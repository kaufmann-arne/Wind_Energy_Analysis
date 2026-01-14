"""
MODEL 0 — FULL PIPELINE (2015–2046, monthly)

Build + forecast the 3 core series and combine them into merchant + EEG revenue:

1) Market price forecast (day-ahead):     P_mkt_hat[t]
   - Drifting-mean AR(1) on log(P + SHIFT), with month seasonality
   - Explicit annual drift g (inflation/demand assumption)

2) Cannibalism / capture factor forecast: CF_hat[t]
   - Regression on log(CF): trend + month seasonality + log(WindMWh) (optional driver)
   - CF_hat = exp(pred), then clipped to historical percentile bounds

3) Curtailment rate forecast:             CR_hat[t]
   - Regression on CR: trend + month seasonality + log(WindMWh) (optional driver)
   - CR_hat clipped to [0, CR_max]

Combine:
  P_wind_mer_hat[t] = P_mkt_hat[t] * CF_hat[t]
  MWh_net_hat[t]    = MWh_gross_park[t] * (1 - CR_hat[t])
  Rev_mer_hat[t]    = MWh_net_hat[t] * P_wind_mer_hat[t]
  P_wind_EEG_hat[t] = max(P_wind_mer_hat[t], Strike[t])
  Rev_EEG_hat[t]    = MWh_net_hat[t] * P_wind_EEG_hat[t]

Inputs you need in the same folder:
- prices_hourly_merged.csv                    (hourly day-ahead prices)
- wind_onshore_hourly_merged.csv              (hourly onshore wind MWh)
- curtailment_wind_monthly.csv                (monthly curtailed wind MWh)  [or quarterly -> set flag]
- eeg_auction_strike_prices.csv               (auction_date + strike_eur_mwh)
- park_gross_mwh_monthly_mock_2015_2046.csv   (date + mwh_gross)  [your mock or real yield]

Outputs:
- model0_forecast_2015_2046_monthly.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# -----------------------------
# CONFIG
# -----------------------------
# Hourly SMARD (merged)
PRICES_HOURLY_FILE = "prices_hourly_merged.csv"
WIND_HOURLY_FILE   = "wind_onshore_hourly_merged.csv"

# If you don't know the exact column names, set them to None and the script will try to auto-detect.
PRICE_TS_COL  = None      # e.g. "date" / "timestamp"
PRICE_COL     = None      # e.g. "price_eur_mwh" or "Germany/Luxembourg [€/MWh] Calculated resolutions"
WIND_TS_COL   = None      # e.g. "date" / "timestamp"
WIND_MWH_COL  = None      # e.g. "wind_onshore_mwh" or "Wind onshore [MWh] Calculated resolutions"

# Market price model (Option 2)
PRICE_SHIFT   = 200.0     # to allow log even with negative prices
ANNUAL_DRIFT  = 0.02      # 2% nominal p.a. (your structural assumption)
ANCHOR_START  = "2020-01-01"
ANCHOR_END    = "2024-12-01"

# Capture factor model
USE_WIND_DRIVER_CF = True
CF_CLIP_PCTL = (1, 99)    # clip CF_hat to historical pctl range

# Curtailment model
CURTAILMENT_FILE = "curtailment_wind_monthly.csv"
CURTAIL_DATE_COL = "date"
CURTAIL_MWH_COL  = "curtailed_wind_mwh"
CURTAIL_IS_QUARTERLY = False
USE_WIND_DRIVER_CR = True
CR_MAX_MARGIN = 0.02      # add small margin to historical max for clipping

# Wind driver projection for future (used if USE_WIND_DRIVER_CF/CR = True)
WIND_GROWTH_ANNUAL = 0.01  # 1% p.a. growth in national wind output proxy (scenario knob)
WIND_SEASON_FROM_YEARS = 5 # use last N years to compute seasonal profile

# EEG strike
EEG_AUCTION_FILE = "eeg_auction_strike_prices.csv"
EEG_AUCTION_DATE_COL = "auction_date"
EEG_STRIKE_EUR_MWH_COL = "strike_eur_mwh"
STRIKE_FORWARD_RULE = "roll24"  # "ffill" or "roll24"

# Park gross MWh monthly (mock or real)
PARK_FILE = "park_gross_mwh_monthly_mock_2015_2046.csv"
PARK_DATE_COL = "date"
PARK_GROSS_COL = "mwh_gross"

# Output
OUTPUT_FILE = "model0_forecast_2015_2046_monthly.csv"

# Forecast horizon
FORECAST_END = "2046-12-01"  # inclusive, month-start


# -----------------------------
# Helpers
# -----------------------------
def to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())

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
    # Try keyword match
    cols = list(df.columns)
    for kw in prefer_keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                # check numeric-ish
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().mean() > 0.5:
                    return c
    # fallback: first numeric-ish column
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            return c
    raise ValueError(f"Could not autodetect numeric column. Columns: {cols}")

def expand_quarterly_to_monthly(df_q: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = df_q.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    df["q"] = df[date_col].dt.to_period("Q")

    rows = []
    for _, r in df.iterrows():
        q_period = r["q"]
        months = pd.period_range(q_period.start_time, q_period.end_time, freq="M")
        v_each = float(r[value_col]) / 3.0
        for m in months:
            rows.append({"date": m.to_timestamp(), value_col: v_each})
    out = pd.DataFrame(rows)
    out["date"] = to_month_start(out["date"])
    out = out.groupby("date", as_index=False)[value_col].sum()
    return out

def build_monthly_strike_series(auction_df: pd.DataFrame, start_month: pd.Timestamp, end_month: pd.Timestamp) -> pd.DataFrame:
    df = auction_df.copy()
    df[EEG_AUCTION_DATE_COL] = pd.to_datetime(df[EEG_AUCTION_DATE_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[EEG_AUCTION_DATE_COL, EEG_STRIKE_EUR_MWH_COL]).copy()

    df["date"] = to_month_start(df[EEG_AUCTION_DATE_COL])
    monthly = df.groupby("date", as_index=False)[EEG_STRIKE_EUR_MWH_COL].mean().rename(
        columns={EEG_STRIKE_EUR_MWH_COL: "strike_eur_mwh"}
    ).sort_values("date")

    if STRIKE_FORWARD_RULE == "roll24":
        monthly["strike_eur_mwh"] = monthly["strike_eur_mwh"].rolling(24, min_periods=1).mean()

    full_months = pd.date_range(start_month, end_month, freq="MS")
    out = pd.DataFrame({"date": full_months}).merge(monthly, on="date", how="left")
    out["strike_eur_mwh"] = out["strike_eur_mwh"].ffill()
    return out

def seasonal_wind_projection(wind_monthly: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Project future national wind MWh proxy:
    - seasonal profile from last WIND_SEASON_FROM_YEARS years (month-of-year factors)
    - level anchored to last 12-month mean
    - optional growth rate WIND_GROWTH_ANNUAL
    """
    wm = wind_monthly.copy().sort_values("date").reset_index(drop=True)
    last_date = wm["date"].max()
    cutoff = last_date - pd.DateOffset(years=WIND_SEASON_FROM_YEARS)
    wm_recent = wm[wm["date"] >= cutoff].copy()
    if len(wm_recent) < 24:
        wm_recent = wm.copy()

    wm_recent["m"] = wm_recent["date"].dt.month
    season = wm_recent.groupby("m")["wind_mwh"].mean()
    season = season / season.mean()  # mean = 1
    season = season.reindex(range(1, 13), fill_value=1.0)

    level = float(wm.tail(12)["wind_mwh"].mean())
    if not np.isfinite(level) or level <= 0:
        level = float(wm["wind_mwh"].median())

    base_months_ahead = ((future_dates.year - last_date.year) * 12 + (future_dates.month - last_date.month)).astype(float)
    growth = (1.0 + WIND_GROWTH_ANNUAL) ** (base_months_ahead / 12.0)

    proj = []
    for d, g in zip(future_dates, growth):
        proj.append(level * float(season.loc[d.month]) * float(g))
    return pd.Series(proj, index=future_dates, name="wind_mwh_proxy_future")


# -----------------------------
# 1) Build monthly panel from hourly price + wind
# -----------------------------
prices = pd.read_csv(PRICES_HOURLY_FILE)
wind   = pd.read_csv(WIND_HOURLY_FILE)

if PRICE_TS_COL is None or PRICE_TS_COL not in prices.columns:
    PRICE_TS_COL = autodetect_datetime_col(prices)
if WIND_TS_COL is None or WIND_TS_COL not in wind.columns:
    WIND_TS_COL = autodetect_datetime_col(wind)

if PRICE_COL is None or PRICE_COL not in prices.columns:
    PRICE_COL = autodetect_numeric_col(prices, prefer_keywords=["eur", "mwh", "price", "Germany", "Luxembourg", "DE"])
if WIND_MWH_COL is None or WIND_MWH_COL not in wind.columns:
    WIND_MWH_COL = autodetect_numeric_col(wind, prefer_keywords=["wind", "onshore", "mwh"])

prices[PRICE_TS_COL] = pd.to_datetime(prices[PRICE_TS_COL], errors="coerce")
wind[WIND_TS_COL] = pd.to_datetime(wind[WIND_TS_COL], errors="coerce")

prices = prices.dropna(subset=[PRICE_TS_COL])
wind   = wind.dropna(subset=[WIND_TS_COL])

prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")
wind[WIND_MWH_COL] = pd.to_numeric(wind[WIND_MWH_COL], errors="coerce")
prices = prices.dropna(subset=[PRICE_COL])
wind   = wind.dropna(subset=[WIND_MWH_COL])

h = prices[[PRICE_TS_COL, PRICE_COL]].merge(
    wind[[WIND_TS_COL, WIND_MWH_COL]],
    left_on=PRICE_TS_COL,
    right_on=WIND_TS_COL,
    how="inner"
).rename(columns={PRICE_TS_COL: "ts", PRICE_COL: "price_eur_mwh", WIND_MWH_COL: "wind_mwh"}).drop(columns=[WIND_TS_COL])

h["date"] = to_month_start(h["ts"])

# Market price monthly
mkt = h.groupby("date", as_index=False)["price_eur_mwh"].mean().rename(columns={"price_eur_mwh": "p_market_eur_mwh"})

# Wind capture price monthly
h["price_times_wind"] = h["price_eur_mwh"] * h["wind_mwh"]
cap = h.groupby("date", as_index=False).agg(
    wind_mwh=("wind_mwh", "sum"),
    price_times_wind=("price_times_wind", "sum"),
)
cap["p_wind_capture_eur_mwh"] = cap["price_times_wind"] / cap["wind_mwh"]
cap = cap[["date", "wind_mwh", "p_wind_capture_eur_mwh"]]

panel = mkt.merge(cap, on="date", how="inner").sort_values("date").reset_index(drop=True)
panel["capture_factor"] = panel["p_wind_capture_eur_mwh"] / panel["p_market_eur_mwh"]

# -----------------------------
# 2) Curtailment rate series (monthly)
# -----------------------------
cur = pd.read_csv(CURTAILMENT_FILE)
cur[CURTAIL_DATE_COL] = pd.to_datetime(cur[CURTAIL_DATE_COL], errors="coerce")
cur[CURTAIL_MWH_COL] = pd.to_numeric(cur[CURTAIL_MWH_COL], errors="coerce")
cur = cur.dropna(subset=[CURTAIL_DATE_COL, CURTAIL_MWH_COL]).copy()
cur = cur.rename(columns={CURTAIL_DATE_COL: "date", CURTAIL_MWH_COL: "curtailed_wind_mwh"})

if CURTAIL_IS_QUARTERLY:
    cur = expand_quarterly_to_monthly(cur, "date", "curtailed_wind_mwh")
else:
    cur["date"] = to_month_start(cur["date"])
    cur = cur.groupby("date", as_index=False)["curtailed_wind_mwh"].sum()

tmp = panel[["date", "wind_mwh"]].rename(columns={"wind_mwh": "actual_wind_mwh"}).merge(cur, on="date", how="left")
tmp["curtailed_wind_mwh"] = tmp["curtailed_wind_mwh"].fillna(0.0)
den = tmp["actual_wind_mwh"] + tmp["curtailed_wind_mwh"]
tmp["curtailment_rate"] = np.where(den > 0, tmp["curtailed_wind_mwh"] / den, 0.0)

panel = panel.merge(tmp[["date", "curtailment_rate"]], on="date", how="left").sort_values("date").reset_index(drop=True)

# -----------------------------
# 3) Build date range (2015–2046) and merge park MWh + strike
# -----------------------------
all_dates = pd.date_range("2015-01-01", FORECAST_END, freq="MS")
base = pd.DataFrame({"date": all_dates})

# Park gross MWh (required for revenue)
park = pd.read_csv(PARK_FILE)
park[PARK_DATE_COL] = to_month_start(pd.to_datetime(park[PARK_DATE_COL], errors="coerce"))
park[PARK_GROSS_COL] = pd.to_numeric(park[PARK_GROSS_COL], errors="coerce")
park = park.dropna(subset=[PARK_DATE_COL, PARK_GROSS_COL]).rename(columns={PARK_DATE_COL: "date", PARK_GROSS_COL: "mwh_gross_park"})
base = base.merge(park[["date", "mwh_gross_park"]], on="date", how="left")

# EEG strike monthly series to horizon
auctions = pd.read_csv(EEG_AUCTION_FILE)
strike = build_monthly_strike_series(auctions, start_month=base["date"].min(), end_month=base["date"].max())
base = base.merge(strike, on="date", how="left")

# -----------------------------
# 4) Market price model (Option 2) — drifting mean AR(1) on log(P + SHIFT)
# -----------------------------
hist = panel.copy()
hist = hist.dropna(subset=["p_market_eur_mwh"]).copy()
hist["p_shift"] = hist["p_market_eur_mwh"] + PRICE_SHIFT
if (hist["p_shift"] <= 0).any():
    raise ValueError("PRICE_SHIFT too small. Increase PRICE_SHIFT so all shifted prices are > 0.")

hist["y"] = np.log(hist["p_shift"])

mask_anchor = (hist["date"] >= pd.Timestamp(ANCHOR_START)) & (hist["date"] <= pd.Timestamp(ANCHOR_END))
if mask_anchor.sum() < 12:
    raise ValueError("Anchor window too short/missing. Adjust ANCHOR_START/ANCHOR_END.")
anchor_date = hist.loc[mask_anchor, "date"].iloc[-1]
anchor_level = float(hist.loc[mask_anchor, "p_shift"].median())
mu0 = float(np.log(anchor_level))

months_from_anchor = (hist["date"].dt.year - anchor_date.year) * 12 + (hist["date"].dt.month - anchor_date.month)
hist["mu"] = mu0 + (months_from_anchor / 12.0) * np.log(1.0 + ANNUAL_DRIFT)
hist["z"] = hist["y"] - hist["mu"]
hist["z_lag"] = hist["z"].shift(1)
hist["month_num"] = hist["date"].dt.month.astype(int)

m_dum = pd.get_dummies(hist["month_num"], prefix="m", drop_first=True).astype(float)
X = pd.concat([hist["z_lag"], m_dum], axis=1)
X = sm.add_constant(X)
mdl_data = pd.concat([hist["z"], X], axis=1).dropna()

mdl_price = sm.OLS(mdl_data["z"].astype(float), mdl_data[X.columns].astype(float)).fit()
phi_p = float(mdl_price.params["z_lag"])
const_p = float(mdl_price.params["const"])
season_p = {k: float(v) for k, v in mdl_price.params.items() if k.startswith("m_")}

# Forecast market price for all dates in base (fill historical with actual where available)
base = base.merge(hist[["date", "p_market_eur_mwh"]], on="date", how="left")

# last observed z for recursion
last_hist_date = hist["date"].max()
z_prev = float(hist.loc[hist["date"] == last_hist_date, "z"].iloc[0])

p_mkt_hat = []
for d in base["date"]:
    if pd.notna(base.loc[base["date"] == d, "p_market_eur_mwh"]).iloc[0]:
        # use actual for in-sample history
        p_mkt_hat.append(float(base.loc[base["date"] == d, "p_market_eur_mwh"].iloc[0]))
        continue

    # forecast month
    months_ahead = (d.year - anchor_date.year) * 12 + (d.month - anchor_date.month)
    mu_t = mu0 + (months_ahead / 12.0) * np.log(1.0 + ANNUAL_DRIFT)
    s_t = season_p.get(f"m_{d.month}", 0.0)
    z_t = const_p + phi_p * z_prev + s_t
    y_t = mu_t + z_t
    p_t = float(np.exp(y_t) - PRICE_SHIFT)

    p_mkt_hat.append(p_t)
    z_prev = z_t

base["p_market_hat"] = p_mkt_hat

# -----------------------------
# 5) Wind driver for CF/CR (project future wind_mwh proxy)
# -----------------------------
wind_monthly = panel[["date", "wind_mwh"]].dropna().copy()
wind_monthly = wind_monthly.sort_values("date").reset_index(drop=True)

future_mask = base["date"] > wind_monthly["date"].max()
wind_future = seasonal_wind_projection(wind_monthly.rename(columns={"wind_mwh": "wind_mwh"}), base.loc[future_mask, "date"])
base = base.merge(wind_monthly.rename(columns={"wind_mwh": "wind_mwh_proxy"}), on="date", how="left")
base.loc[future_mask, "wind_mwh_proxy"] = wind_future.values

# -----------------------------
# 6) Capture factor model: log(CF) ~ trend + season + log(wind_mwh_proxy)
# -----------------------------
cf_hist = panel.dropna(subset=["capture_factor", "wind_mwh"]).copy()
cf_hist = cf_hist[(cf_hist["capture_factor"] > 0) & np.isfinite(cf_hist["capture_factor"])].copy()

cf_hist["u"] = np.log(cf_hist["capture_factor"])
cf_hist["trend"] = np.arange(len(cf_hist), dtype=float)
cf_hist["month_num"] = cf_hist["date"].dt.month.astype(int)
cf_dum = pd.get_dummies(cf_hist["month_num"], prefix="m", drop_first=True).astype(float)

X_cf_parts = [cf_hist["trend"]]
if USE_WIND_DRIVER_CF:
    cf_hist["log_wind"] = np.log(cf_hist["wind_mwh"].clip(lower=1.0))
    X_cf_parts.append(cf_hist["log_wind"])
X_cf = pd.concat(X_cf_parts + [cf_dum], axis=1)
X_cf = sm.add_constant(X_cf)
mdl_cf_data = pd.concat([cf_hist["u"], X_cf], axis=1).dropna()
mdl_cf = sm.OLS(mdl_cf_data["u"].astype(float), mdl_cf_data[X_cf.columns].astype(float)).fit()

# CF bounds from historical percentiles
lo_p, hi_p = CF_CLIP_PCTL
cf_lo = float(np.percentile(cf_hist["capture_factor"].values, lo_p))
cf_hi = float(np.percentile(cf_hist["capture_factor"].values, hi_p))

# Forecast CF for all dates in base
# Build feature frame aligned to base dates
cf_feat = pd.DataFrame({"date": base["date"]})
cf_feat["month_num"] = cf_feat["date"].dt.month.astype(int)
cf_feat["trend"] = np.arange(len(cf_feat), dtype=float)  # simple continuation
cf_feat_dum = pd.get_dummies(cf_feat["month_num"], prefix="m", drop_first=True).astype(float)

X_cf_f_parts = [cf_feat["trend"]]
if USE_WIND_DRIVER_CF:
    X_cf_f_parts.append(np.log(base["wind_mwh_proxy"].clip(lower=1.0)).rename("log_wind"))

X_cf_f = pd.concat(X_cf_f_parts + [cf_feat_dum], axis=1)
# align columns to training design (add missing dummy cols)
for col in mdl_cf.params.index:
    if col.startswith("m_") and col not in X_cf_f.columns:
        X_cf_f[col] = 0.0
X_cf_f = sm.add_constant(X_cf_f, has_constant="add")
X_cf_f = X_cf_f[mdl_cf.params.index]  # same order

u_hat = (X_cf_f.astype(float) @ mdl_cf.params.astype(float)).values
cf_hat = np.exp(u_hat)
cf_hat = np.clip(cf_hat, cf_lo, cf_hi)

base["cf_hat"] = cf_hat

# -----------------------------
# 7) Curtailment model: CR ~ trend + season + log(wind_mwh_proxy)
# -----------------------------
cr_hist = panel.dropna(subset=["curtailment_rate", "wind_mwh"]).copy()
cr_hist["cr"] = cr_hist["curtailment_rate"].clip(lower=0.0)
cr_hist["trend"] = np.arange(len(cr_hist), dtype=float)
cr_hist["month_num"] = cr_hist["date"].dt.month.astype(int)
cr_dum = pd.get_dummies(cr_hist["month_num"], prefix="m", drop_first=True).astype(float)

X_cr_parts = [cr_hist["trend"]]
if USE_WIND_DRIVER_CR:
    cr_hist["log_wind"] = np.log(cr_hist["wind_mwh"].clip(lower=1.0))
    X_cr_parts.append(cr_hist["log_wind"])
X_cr = pd.concat(X_cr_parts + [cr_dum], axis=1)
X_cr = sm.add_constant(X_cr)
mdl_cr_data = pd.concat([cr_hist["cr"], X_cr], axis=1).dropna()
mdl_cr = sm.OLS(mdl_cr_data["cr"].astype(float), mdl_cr_data[X_cr.columns].astype(float)).fit()

cr_max_hist = float(cr_hist["cr"].max())
cr_cap = min(1.0, cr_max_hist + CR_MAX_MARGIN)

# Forecast CR for all dates
cr_feat = pd.DataFrame({"date": base["date"]})
cr_feat["month_num"] = cr_feat["date"].dt.month.astype(int)
cr_feat["trend"] = np.arange(len(cr_feat), dtype=float)
cr_feat_dum = pd.get_dummies(cr_feat["month_num"], prefix="m", drop_first=True).astype(float)

X_cr_f_parts = [cr_feat["trend"]]
if USE_WIND_DRIVER_CR:
    X_cr_f_parts.append(np.log(base["wind_mwh_proxy"].clip(lower=1.0)).rename("log_wind"))
X_cr_f = pd.concat(X_cr_f_parts + [cr_feat_dum], axis=1)

for col in mdl_cr.params.index:
    if col.startswith("m_") and col not in X_cr_f.columns:
        X_cr_f[col] = 0.0
X_cr_f = sm.add_constant(X_cr_f, has_constant="add")
X_cr_f = X_cr_f[mdl_cr.params.index]

cr_hat = (X_cr_f.astype(float) @ mdl_cr.params.astype(float)).values
cr_hat = np.clip(cr_hat, 0.0, cr_cap)

base["cr_hat"] = cr_hat

# -----------------------------
# 8) Combine into prices + revenues
# -----------------------------
base["p_wind_merchant_hat"] = base["p_market_hat"] * base["cf_hat"]
base["availability_factor_hat"] = 1.0 - base["cr_hat"]
base["mwh_net_hat"] = base["mwh_gross_park"] * base["availability_factor_hat"]

base["rev_merchant_hat_eur"] = base["mwh_net_hat"] * base["p_wind_merchant_hat"]

base["p_wind_eeg_hat"] = np.maximum(base["p_wind_merchant_hat"], base["strike_eur_mwh"])
base["rev_eeg_hat_eur"] = base["mwh_net_hat"] * base["p_wind_eeg_hat"]

# Keep a tidy column set
out_cols = [
    "date",
    "mwh_gross_park",
    "wind_mwh_proxy",
    "p_market_hat",
    "cf_hat",
    "cr_hat",
    "p_wind_merchant_hat",
    "strike_eur_mwh",
    "p_wind_eeg_hat",
    "mwh_net_hat",
    "rev_merchant_hat_eur",
    "rev_eeg_hat_eur",
]
out = base[out_cols].copy()
out.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Detected columns -> PRICE_TS_COL='{PRICE_TS_COL}', PRICE_COL='{PRICE_COL}', WIND_TS_COL='{WIND_TS_COL}', WIND_MWH_COL='{WIND_MWH_COL}'")
print(f"Saved: {OUTPUT_FILE}")
print(f"Market drift g={ANNUAL_DRIFT:.2%}, CF clipped to p{CF_CLIP_PCTL[0]}–p{CF_CLIP_PCTL[1]} ({cf_lo:.3f}..{cf_hi:.3f}), CR capped at {cr_cap:.3f}")
