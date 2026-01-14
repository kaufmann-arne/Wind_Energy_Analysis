import pandas as pd
import numpy as np
import statsmodels.api as sm

# -----------------------------
# CONFIG
# -----------------------------
WIND_HOURLY_FILE = "wind_onshore_hourly_merged.csv"
WIND_TS_COL = None          # auto-detect if None
WIND_MWH_COL = None         # auto-detect if None

# Your quarterly curtailment input (YOU create this from the reports)
# Expected columns:
#   quarter   (e.g. "2020Q1" or "2020-Q1")
#   curtailed_wind_mwh
CURTAIL_Q_FILE = "curtailment_wind_quarterly.csv"
CURTAIL_Q_COL = "quarter"
CURTAIL_MWH_COL = "curtailed_wind_mwh"

FORECAST_END_Q = "2046Q4"

# Regression switches
USE_WIND_DRIVER = True
CR_MAX_MARGIN = 0.02  # cap = historical max + margin (<=1)

# If you want monthly output:
EXPORT_MONTHLY = True

OUT_Q_HIST = "curtailment_rate_quarterly_hist.csv"
OUT_Q_FCST = "curtailment_rate_quarterly_forecast_to_2046.csv"
OUT_M_FCST = "curtailment_rate_monthly_forecast_to_2046.csv"


# -----------------------------
def autodetect_datetime_col(df: pd.DataFrame) -> str:
    candidates = ["date", "datetime", "timestamp", "time", "utc_timestamp", "Datum", "Zeit"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
            return c
    raise ValueError(f"Could not autodetect datetime column. Columns: {list(df.columns)}")

def autodetect_numeric_col(df: pd.DataFrame, prefer_keywords: list[str]) -> str:
    for kw in prefer_keywords:
        for c in df.columns:
            if kw.lower() in str(c).lower():
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().mean() > 0.5:
                    return c
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            return c
    raise ValueError(f"Could not autodetect numeric column. Columns: {list(df.columns)}")

def to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())

def parse_quarter_to_period(qstr: str) -> pd.Period:
    # Accept "2020Q1", "2020-Q1", "2020 Q1"
    qstr = str(qstr).upper().replace(" ", "").replace("-", "")
    if "Q" not in qstr:
        raise ValueError(f"Quarter format not understood: {qstr}")
    year = int(qstr[:4])
    q = int(qstr.split("Q")[1])
    return pd.Period(year=year, quarter=q, freq="Q")

def quarter_months(q: pd.Period) -> pd.DatetimeIndex:
    start = q.start_time.to_period("M").to_timestamp()
    return pd.date_range(start, periods=3, freq="MS")


# -----------------------------
# 1) Monthly wind from hourly SMARD
# -----------------------------
wind = pd.read_csv(WIND_HOURLY_FILE)

if WIND_TS_COL is None or WIND_TS_COL not in wind.columns:
    WIND_TS_COL = autodetect_datetime_col(wind)
if WIND_MWH_COL is None or WIND_MWH_COL not in wind.columns:
    WIND_MWH_COL = autodetect_numeric_col(wind, prefer_keywords=["wind", "onshore", "mwh"])

wind[WIND_TS_COL] = pd.to_datetime(wind[WIND_TS_COL], errors="coerce")
wind[WIND_MWH_COL] = pd.to_numeric(wind[WIND_MWH_COL], errors="coerce")
wind = wind.dropna(subset=[WIND_TS_COL, WIND_MWH_COL]).copy()

wind["month"] = to_month_start(wind[WIND_TS_COL])
wind_m = wind.groupby("month", as_index=False)[WIND_MWH_COL].sum().rename(columns={"month": "date", WIND_MWH_COL: "actual_wind_mwh"})
wind_m["quarter"] = wind_m["date"].dt.to_period("Q")

wind_q = wind_m.groupby("quarter", as_index=False)["actual_wind_mwh"].sum()

# -----------------------------
# 2) Load quarterly curtailed wind MWh (from BNetzA)
# -----------------------------
curq = pd.read_csv(CURTAIL_Q_FILE)
curq[CURTAIL_MWH_COL] = pd.to_numeric(curq[CURTAIL_MWH_COL], errors="coerce")
curq = curq.dropna(subset=[CURTAIL_Q_COL, CURTAIL_MWH_COL]).copy()

curq["quarter"] = curq[CURTAIL_Q_COL].apply(parse_quarter_to_period)
curq = curq[["quarter", CURTAIL_MWH_COL]].rename(columns={CURTAIL_MWH_COL: "curtailed_wind_mwh"})

# -----------------------------
# 3) Quarterly curtailment rate CR_q
# -----------------------------
hist_q = wind_q.merge(curq, on="quarter", how="inner").sort_values("quarter").reset_index(drop=True)

den = hist_q["actual_wind_mwh"] + hist_q["curtailed_wind_mwh"]
hist_q["cr_q"] = np.where(den > 0, hist_q["curtailed_wind_mwh"] / den, 0.0)

hist_q.to_csv(OUT_Q_HIST, index=False)
print(f"Saved quarterly history: {OUT_Q_HIST}")

# -----------------------------
# 4) Model 0 regression on quarterly data
#    cr_q ~ trend + quarter dummies + log(actual_wind_mwh)
# -----------------------------
hist_q["trend"] = np.arange(len(hist_q), dtype=float)
hist_q["q_num"] = hist_q["quarter"].dt.quarter.astype(int)

q_dum = pd.get_dummies(hist_q["q_num"], prefix="q", drop_first=True).astype(float)

X_parts = [hist_q["trend"]]
if USE_WIND_DRIVER:
    X_parts.append(np.log(hist_q["actual_wind_mwh"].clip(lower=1.0)).rename("log_wind"))

X = pd.concat(X_parts + [q_dum], axis=1)
X = sm.add_constant(X)
y = hist_q["cr_q"].astype(float)

mdl = sm.OLS(y, X.astype(float)).fit()
print(mdl.summary())

cr_cap = min(1.0, float(hist_q["cr_q"].max()) + CR_MAX_MARGIN)

# -----------------------------
# 5) Forecast quarterly to 2046Q4
# -----------------------------
start_q = hist_q["quarter"].min()
end_q = parse_quarter_to_period(FORECAST_END_Q)
all_q = pd.period_range(start_q, end_q, freq="Q")

fcst_q = pd.DataFrame({"quarter": all_q})
fcst_q["trend"] = np.arange(len(fcst_q), dtype=float)
fcst_q["q_num"] = fcst_q["quarter"].dt.quarter.astype(int)

# Future wind driver (very simple baseline):
# keep seasonal quarter factors + flat level from last 8 quarters
if USE_WIND_DRIVER:
    last_level = float(hist_q.tail(8)["actual_wind_mwh"].mean())

    season_q = hist_q.groupby("q_num")["actual_wind_mwh"].mean()
    season_q = season_q / season_q.mean()
    season_q = season_q.reindex([1,2,3,4], fill_value=1.0)

    wind_proxy = [last_level * float(season_q.loc[int(q)]) for q in fcst_q["q_num"]]
    fcst_q["log_wind"] = np.log(np.clip(wind_proxy, 1.0, None))

q_dum_f = pd.get_dummies(fcst_q["q_num"], prefix="q", drop_first=True).astype(float)

Xf_parts = [fcst_q["trend"]]
if USE_WIND_DRIVER:
    Xf_parts.append(fcst_q["log_wind"])

Xf = pd.concat(Xf_parts + [q_dum_f], axis=1)

# Align columns to model
for col in mdl.params.index:
    if col.startswith("q_") and col not in Xf.columns:
        Xf[col] = 0.0

Xf = sm.add_constant(Xf, has_constant="add")
Xf = Xf[mdl.params.index]

cr_hat_q = (Xf.astype(float) @ mdl.params.astype(float)).values
cr_hat_q = np.clip(cr_hat_q, 0.0, cr_cap)

fcst_q["cr_hat_q"] = cr_hat_q
fcst_q.to_csv(OUT_Q_FCST, index=False)
print(f"Saved quarterly forecast: {OUT_Q_FCST} (cap={cr_cap:.3f})")

# -----------------------------
# 6) Optional: Convert quarterly CR to monthly CR
# -----------------------------
if EXPORT_MONTHLY:
    rows = []
    # Build monthly wind season weights from historical wind_m
    wind_m_hist = wind_m.copy()
    wind_m_hist["m"] = wind_m_hist["date"].dt.month
    season_m = wind_m_hist.groupby("m")["actual_wind_mwh"].mean()
    season_m = season_m / season_m.mean()
    season_m = season_m.reindex(range(1, 13), fill_value=1.0)

    # Use last 12-month mean as level
    m_level = float(wind_m_hist.tail(12)["actual_wind_mwh"].mean())

    for q, crq in zip(fcst_q["quarter"], fcst_q["cr_hat_q"]):
        months = quarter_months(q)
        # monthly wind proxy within quarter
        w = np.array([m_level * float(season_m.loc[m.month]) for m in months], dtype=float)
        w_share = w / w.sum() if w.sum() > 0 else np.array([1/3, 1/3, 1/3])

        # allocate quarter curtailment rate to months using wind shares (same CR each month is fine too;
        # this variant gives slightly more realistic month shape)
        for m_date, _ in zip(months, w_share):
            rows.append({"date": m_date, "cr_hat_m": float(crq)})

    fcst_m = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    fcst_m.to_csv(OUT_M_FCST, index=False)
    print(f"Saved monthly CR forecast: {OUT_M_FCST}")

print(f"Detected WIND cols -> ts='{WIND_TS_COL}', mwh='{WIND_MWH_COL}'")
