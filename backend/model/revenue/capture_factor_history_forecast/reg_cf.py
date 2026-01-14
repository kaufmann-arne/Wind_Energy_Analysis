"""
Scenario B2: Capture factor (CF) regression + forecast using scaled historical wind.

We model:
  z_t = log(CF_t)
  z_t = const + phi*z_{t-1} + season(month) + beta*log(wind_mwh_t) + error

Then we forecast CF for 20 years by:
- building future wind_mwh_t by scaling a repeating monthly wind pattern
  from a recent reference window, with an annual growth rate per scenario
- recursively forecasting z_t with AR(1) dynamics

Inputs (monthly, historical):
  capture_factor_monthly_historical.csv
    columns: month, p_market_eur_mwh, wind_mwh, p_wind_capture_eur_mwh, capture_factor

Outputs:
  capture_factor_forecast_b2.csv
    month, wind_mwh_scen_*, cf_scen_*, p_wind_merchant_scen_* (optional)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "capture_factor_monthly_historical.csv"  # created by your CF history script
DATE_COL = "month"
CF_COL = "capture_factor"
WIND_COL = "wind_mwh"

FORECAST_YEARS = 20

# Reference window used to build the repeating seasonal wind pattern (monthly averages)
REF_START = "2020-01-01"
REF_END   = "2024-12-01"

# Wind growth scenarios (annual growth of national wind generation proxy)
WIND_GROWTH_SCEN = {
    "low":  0.005,  # +0.5% p.a.
    "base": 0.015,  # +1.5% p.a.
    "high": 0.030,  # +3.0% p.a.
}

# CF forecast guardrails (applied after exp() back-transform)
CF_FLOOR = 0.55
CF_CAP   = 1.15

OUTPUT_FILE = "capture_factor_forecast_b2.csv"

# -----------------------------
def to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s)
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())

# -----------------------------
# 1) Load historical monthly CF
# -----------------------------
df = pd.read_csv(INPUT_FILE, parse_dates=[DATE_COL])
df[DATE_COL] = to_month_start(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Basic checks
df[CF_COL] = pd.to_numeric(df[CF_COL], errors="coerce")
df[WIND_COL] = pd.to_numeric(df[WIND_COL], errors="coerce")
df = df.dropna(subset=[CF_COL, WIND_COL])

# CF must be positive for log
df = df[df[CF_COL] > 0].copy()

# -----------------------------
# 2) Build regression dataset: z_t = log(CF_t)
# -----------------------------
df["z"] = np.log(df[CF_COL].values.astype(float))
df["z_lag"] = df["z"].shift(1)
df["month_num"] = df[DATE_COL].dt.month.astype(int)

# Seasonality via month dummies
month_dummies = pd.get_dummies(df["month_num"], prefix="m", drop_first=True).astype(float)

# Wind driver in logs (avoid log(0))
df["log_wind"] = np.log(np.maximum(df[WIND_COL].values.astype(float), 1.0))

X = pd.concat([df["z_lag"], df["log_wind"], month_dummies], axis=1)
X = sm.add_constant(X)
reg = pd.concat([df["z"], X], axis=1).dropna()

model = sm.OLS(reg["z"].astype(float), reg[X.columns].astype(float)).fit()
print(model.summary())

params = model.params
const = float(params["const"])
phi = float(params["z_lag"])
beta_wind = float(params["log_wind"])
season_params = {k: float(v) for k, v in params.items() if k.startswith("m_")}

# stability clamp (optional but sensible)
phi = max(min(phi, 0.98), 0.0)

print("\n--- Fitted ---")
print(f"phi={phi:.4f}, beta_wind={beta_wind:.4f}, CF bounds [{CF_FLOOR}, {CF_CAP}]")

# -----------------------------
# 3) Build a repeating monthly wind pattern from the reference window
# -----------------------------
mask_ref = (df[DATE_COL] >= pd.Timestamp(REF_START)) & (df[DATE_COL] <= pd.Timestamp(REF_END))
if mask_ref.sum() < 24:
    raise ValueError("Reference window too short/missing. Adjust REF_START/REF_END.")

ref = df.loc[mask_ref, [DATE_COL, WIND_COL]].copy()
ref["m"] = ref[DATE_COL].dt.month

# monthly seasonal wind pattern (average MWh for each month-of-year)
wind_seasonal = ref.groupby("m")[WIND_COL].mean().to_dict()

# anchor wind level: mean in ref window
wind_anchor = float(ref[WIND_COL].mean())
anchor_date = df.loc[mask_ref, DATE_COL].iloc[-1]

print(f"Wind anchor (mean {REF_START}..{REF_END}): {wind_anchor:,.0f} MWh/month (anchor date {anchor_date.date()})")

# -----------------------------
# 4) Forecast horizon dates
# -----------------------------
last_date = df[DATE_COL].max()
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=FORECAST_YEARS * 12, freq="MS")

# Starting state: last observed z
z_prev = float(df["z"].iloc[-1])

# -----------------------------
# 5) Forecast CF under each wind-growth scenario
# -----------------------------
out = pd.DataFrame({DATE_COL: pd.concat([df[DATE_COL], pd.Series(future_dates)], ignore_index=True)})
out["cf_hist"] = pd.concat([df[CF_COL], pd.Series([np.nan] * len(future_dates))], ignore_index=True)
out["wind_mwh_hist"] = pd.concat([df[WIND_COL], pd.Series([np.nan] * len(future_dates))], ignore_index=True)

for scen, g in WIND_GROWTH_SCEN.items():
    # Build future wind series: seasonal shape * growth factor
    wind_future = []
    for d in future_dates:
        months_from_anchor = (d.year - anchor_date.year) * 12 + (d.month - anchor_date.month)
        growth_factor = (1.0 + g) ** (months_from_anchor / 12.0)

        # seasonal baseline for month-of-year
        base_m = float(wind_seasonal[int(d.month)])
        # rescale so the mean matches wind_anchor
        # (seasonal dict already in MWh, but this keeps growth around the anchor)
        w = base_m * growth_factor
        wind_future.append(w)

    wind_future = np.array(wind_future, dtype=float)
    log_wind_future = np.log(np.maximum(wind_future, 1.0))

    # Forecast z_t recursively (deterministic P50 path)
    z_path = np.zeros(len(future_dates), dtype=float)
    z_prev_s = z_prev

    for i, d in enumerate(future_dates):
        season_effect = season_params.get(f"m_{d.month}", 0.0)  # month 1 baseline 0.0
        z_now = const + phi * z_prev_s + beta_wind * log_wind_future[i] + season_effect
        z_path[i] = z_now
        z_prev_s = z_now

    cf_future = np.exp(z_path)

    # Apply guardrails
    cf_future = np.clip(cf_future, CF_FLOOR, CF_CAP)

    # Store
    out[f"wind_mwh_{scen}"] = np.nan
    out[f"cf_{scen}"] = np.nan

    future_mask = out[DATE_COL].isin(future_dates)
    out.loc[future_mask, f"wind_mwh_{scen}"] = wind_future
    out.loc[future_mask, f"cf_{scen}"] = cf_future

# Save
out.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved CF forecast (Scenario B2) to: {OUTPUT_FILE}")
