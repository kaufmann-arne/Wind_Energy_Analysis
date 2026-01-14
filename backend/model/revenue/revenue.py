import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — FILES
# ============================================================
PARK_MWH_FILE = "park_gross_mwh_monthly_mock_2015_2046.csv"
PARK_DATE_COL = "date"
PARK_MWH_COL  = "mwh_gross"          # adjust if needed

MARKET_FILE = "day_ahead_market_prices/market_price_forecast_20y_monthly.csv"
MARKET_DATE_COL = "date"
MARKET_PRICE_COL = "price_eur_mwh_forecast"

CF_FILE = "capture_factor_history_forecast/capture_factor_forecast_b2.csv"
CF_DATE_COL = "month"
CF_PREF_COLS = ["cf_base", "cf_low", "cf_high", "cf_hist"]

CURTAIL_FILE = "curltailment_rate_per_tso/curtailment_forecast_quarterly_by_tso_2021_2046.csv"
CURTAIL_Q_COL = "quarter"
CURTAIL_TSO_COL = "TSO"
CURTAIL_CR_COL = "cr_hat"

EEG_FILE = "eeg/eeg_strike_forecast_monthly.csv"
EEG_DATE_COL = "date"
EEG_STRIKE_COL = "eeg_strike_used"

OUTPUT_MONTHLY = "windpark_revenue_forecast_20y_monthly.csv"
OUTPUT_SUMMARY = "windpark_revenue_summary.csv"
OUTPUT_PNG = "windpark_revenue_over_time.png"

# ============================================================
# CONFIG — USER INPUTS
# ============================================================
TSO_MAP = {
    0: "50Hertz",
    1: "TenneT",
    2: "Amprion",
    3: "TransnetBW",
}
TSO_ID = 0  # <-- set 0..3

EEG_ON = 1  # 1 = EEG supported, 0 = merchant only
MANUAL_EEG_STRIKE = None  # e.g. 75.0 if you want to override
COD_DATE = "2026-01-01"
FORECAST_MONTHS = 20 * 12

CF_CLIP = (0.50, 1.05)
CR_CLIP = (0.00, 0.30)

# ============================================================
# HELPERS
# ============================================================
def to_month_start(s):
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())

def quarter_to_month_starts(q: pd.Period):
    start = q.start_time.to_period("M").to_timestamp()
    return pd.date_range(start, periods=3, freq="MS")

def pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}. Found: {list(df.columns)}")

def ffill_asof(series_df, date_col, value_col, target_dates):
    s = series_df[[date_col, value_col]].dropna().sort_values(date_col)
    return pd.merge_asof(
        pd.DataFrame({date_col: target_dates}).sort_values(date_col),
        s,
        on=date_col,
        direction="backward",
    )[value_col].values

# ============================================================
# LOAD DATA
# ============================================================
# Park MWh
park = pd.read_csv(PARK_MWH_FILE)
park[PARK_DATE_COL] = to_month_start(park[PARK_DATE_COL])
park[PARK_MWH_COL]  = pd.to_numeric(park[PARK_MWH_COL], errors="coerce")
park = park.dropna(subset=[PARK_DATE_COL, PARK_MWH_COL]).copy()
park = park.rename(columns={PARK_DATE_COL: "date", PARK_MWH_COL: "mwh_gross_park"})

# Market price
mkt = pd.read_csv(MARKET_FILE, parse_dates=[MARKET_DATE_COL])
mkt[MARKET_DATE_COL] = to_month_start(mkt[MARKET_DATE_COL])
mkt[MARKET_PRICE_COL] = pd.to_numeric(mkt[MARKET_PRICE_COL], errors="coerce")
mkt = mkt.dropna(subset=[MARKET_DATE_COL, MARKET_PRICE_COL]).copy()
mkt = mkt.rename(columns={MARKET_DATE_COL: "date", MARKET_PRICE_COL: "p_market"})

# Capture factor
cf = pd.read_csv(CF_FILE)
cf[CF_DATE_COL] = to_month_start(cf[CF_DATE_COL])
cf_col = pick_first_existing_col(cf, CF_PREF_COLS)
cf[cf_col] = pd.to_numeric(cf[cf_col], errors="coerce")
cf = cf.rename(columns={CF_DATE_COL: "date", cf_col: "cf_raw"}).sort_values("date")

# Curtailment by TSO (quarterly)
curt = pd.read_csv(CURTAIL_FILE)
curt[CURTAIL_Q_COL] = pd.PeriodIndex(curt[CURTAIL_Q_COL].astype(str), freq="Q")
curt[CURTAIL_CR_COL] = pd.to_numeric(curt[CURTAIL_CR_COL], errors="coerce")
curt = curt.dropna(subset=[CURTAIL_Q_COL, CURTAIL_TSO_COL, CURTAIL_CR_COL]).copy()

tso_name = TSO_MAP.get(int(TSO_ID))
if tso_name is None:
    raise ValueError(f"TSO_ID must be one of {list(TSO_MAP.keys())}")
curt = curt[curt[CURTAIL_TSO_COL] == tso_name].copy()

# Expand quarterly CR to monthly CR
rows = []
for _, r in curt.iterrows():
    q = r[CURTAIL_Q_COL]
    for m in quarter_to_month_starts(q):
        rows.append({"date": pd.Timestamp(m), "cr_raw": float(r[CURTAIL_CR_COL])})
cr_m = pd.DataFrame(rows).groupby("date", as_index=False)["cr_raw"].mean().sort_values("date")

# EEG strike series (used only to pick COD strike if MANUAL_EEG_STRIKE is None)
eeg = pd.read_csv(EEG_FILE, parse_dates=[EEG_DATE_COL])
eeg[EEG_DATE_COL] = to_month_start(eeg[EEG_DATE_COL])
eeg[EEG_STRIKE_COL] = pd.to_numeric(eeg[EEG_STRIKE_COL], errors="coerce")
eeg = eeg.dropna(subset=[EEG_DATE_COL, EEG_STRIKE_COL]).copy()
eeg = eeg.rename(columns={EEG_DATE_COL: "date", EEG_STRIKE_COL: "eeg_strike_series"}).sort_values("date")

# ============================================================
# BUILD MONTHLY INDEX (20y from COD)
# ============================================================
cod = pd.Timestamp(COD_DATE)
cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

dates = pd.date_range(cod, periods=FORECAST_MONTHS, freq="MS")
base = pd.DataFrame({"date": dates})

# Merge market and park
base = base.merge(mkt, on="date", how="left")
if base["p_market"].isna().any():
    missing = base.loc[base["p_market"].isna(), "date"].min()
    raise ValueError(f"Market price missing in horizon. First missing month: {missing.date()}")

base = base.merge(park, on="date", how="left")
if base["mwh_gross_park"].isna().any():
    missing = base.loc[base["mwh_gross_park"].isna(), "date"].min()
    raise ValueError(f"Park gross MWh missing in horizon. First missing month: {missing.date()}")

# Capture factor (asof fill)
base["cf"] = ffill_asof(cf, "date", "cf_raw", base["date"])
base["cf"] = np.clip(base["cf"], CF_CLIP[0], CF_CLIP[1])

# Curtailment (merge, then fill)
base = base.merge(cr_m, on="date", how="left")
base["cr_raw"] = base["cr_raw"].ffill().bfill()
base["cr"] = np.clip(base["cr_raw"], CR_CLIP[0], CR_CLIP[1])

# ============================================================
# EEG STRIKE FIXED AT COD FOR 20 YEARS
# ============================================================
if EEG_ON == 1:
    if MANUAL_EEG_STRIKE is not None:
        strike_cod = float(MANUAL_EEG_STRIKE)
        strike_source = "manual"
    else:
        strike_cod_arr = ffill_asof(eeg, "date", "eeg_strike_series", pd.Series([cod]))
        strike_cod = float(strike_cod_arr[0]) if np.isfinite(strike_cod_arr[0]) else np.nan
        if not np.isfinite(strike_cod):
            raise ValueError(
                "Could not determine EEG strike at/before COD from eeg_strike_forecast_monthly.csv. "
                "Set MANUAL_EEG_STRIKE or ensure EEG file contains values <= COD."
            )
        strike_source = "from_series_asof"

    base["eeg_on"] = 1
    base["strike_cod_fixed"] = strike_cod
    base["strike_source"] = strike_source
else:
    base["eeg_on"] = 0
    base["strike_cod_fixed"] = np.nan
    base["strike_source"] = "off"

# ============================================================
# REVENUE CALCULATION
# ============================================================
base["mwh_delivered"] = base["mwh_gross_park"] * (1.0 - base["cr"])
base["p_wind_merchant"] = base["p_market"] * base["cf"]

if EEG_ON == 1:
    base["p_wind_realised"] = np.maximum(base["p_wind_merchant"], base["strike_cod_fixed"])
else:
    base["p_wind_realised"] = base["p_wind_merchant"]

base["revenue_eur"] = base["mwh_delivered"] * base["p_wind_realised"]

if EEG_ON == 1:
    base["eeg_premium_eur_per_mwh"] = np.maximum(0.0, base["strike_cod_fixed"] - base["p_wind_merchant"])
    base["eeg_premium_eur"] = base["eeg_premium_eur_per_mwh"] * base["mwh_delivered"]
else:
    base["eeg_premium_eur_per_mwh"] = 0.0
    base["eeg_premium_eur"] = 0.0

# ============================================================
# SAVE MONTHLY OUTPUT
# ============================================================
cols = [
    "date",
    "mwh_gross_park", "cr", "mwh_delivered",
    "p_market", "cf", "p_wind_merchant",
    "eeg_on", "strike_cod_fixed", "p_wind_realised",
    "revenue_eur", "eeg_premium_eur",
    "strike_source",
]
base[cols].to_csv(OUTPUT_MONTHLY, index=False)

# ============================================================
# SUMMARY + CHART
# ============================================================
total_revenue = float(base["revenue_eur"].sum())
avg_annual_revenue = total_revenue / 20.0

annual = base.copy()
annual["year"] = annual["date"].dt.year
annual_sum = annual.groupby("year", as_index=False)["revenue_eur"].sum()
annual_sum = annual_sum.rename(columns={"revenue_eur": "revenue_eur_year"})

summary = pd.DataFrame({
    "metric": [
        "TSO",
        "EEG_ON",
        "COD",
        "EEG_strike_fixed_eur_mwh",
        "Total_revenue_20y_eur",
        "Average_annual_revenue_eur",
    ],
    "value": [
        tso_name,
        int(EEG_ON),
        str(cod.date()),
        float(base["strike_cod_fixed"].iloc[0]) if EEG_ON == 1 else np.nan,
        total_revenue,
        avg_annual_revenue,
    ],
})
summary.to_csv(OUTPUT_SUMMARY, index=False)

# Chart: monthly revenue + 12-month rolling mean
plot_df = base[["date", "revenue_eur"]].copy()
plot_df["revenue_rolling_12m"] = plot_df["revenue_eur"].rolling(12, min_periods=1).mean()

plt.figure()
plt.plot(plot_df["date"], plot_df["revenue_eur"], label="Monthly revenue")
plt.plot(plot_df["date"], plot_df["revenue_rolling_12m"], label="Rolling 12M average")
plt.xlabel("Date")
plt.ylabel("Revenue (€)")
plt.title("Wind park revenue forecast (monthly)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)

print("Saved monthly output:", OUTPUT_MONTHLY)
print("Saved summary:", OUTPUT_SUMMARY)
print("Saved chart:", OUTPUT_PNG)
print(f"TOTAL REVENUE (20y): {total_revenue:,.0f} €")
print(f"AVG ANNUAL REVENUE:  {avg_annual_revenue:,.0f} €")
print(f"TSO selected: {tso_name} (TSO_ID={TSO_ID})")
if EEG_ON == 1:
    print(f"EEG strike fixed at COD {cod.date()}: {base['strike_cod_fixed'].iloc[0]:.2f} €/MWh ({base['strike_source'].iloc[0]})")
