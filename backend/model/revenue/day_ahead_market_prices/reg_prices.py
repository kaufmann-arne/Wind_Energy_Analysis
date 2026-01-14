"""
Market price forecast (Germany day-ahead monthly average) with:

Option 2: AR(1) dynamics around a user-chosen drifting mean (anchored),
plus monthly seasonality.

Key idea (log-space, with a positive shift):
  y_t = log(P_t + SHIFT)

Drifting mean:
  mu_t = log(P_anchor + SHIFT) + (months_from_anchor/12) * log(1 + g_annual)

Forecast recursion:
  y_t = mu_t + phi * (y_{t-1} - mu_{t-1}) + season(month_t)

This gives:
- Mean reversion (AR(1))
- Explicit upward drift (inflation/demand) via g_annual
- Seasonal pattern
- Simple + explainable

Outputs:
- market_price_monthly_history.csv (monthly avg from hourly)
- market_price_forecast_20y_monthly.csv (20y monthly forecast, P50 path)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "prices_hourly_merged.csv"

# Leave DATE_COL=None to auto-detect; or set explicitly if you know it
DATE_COL = None

# Set to your actual day-ahead price column in the CSV
PRICE_COL = "price_eur_mwh"   # <-- CHANGE if needed

FORECAST_YEARS = 20

# Shift to handle negative prices before log()
PRICE_SHIFT = 200.0  # increase if you still see non-positive after shifting

# Drift assumption (nominal annual growth)
ANNUAL_DRIFT = 0.01  # 2% p.a. (set 0.03 for 3%, etc.)

# Anchor window: take median level here as "starting point" for drift
ANCHOR_START = "2015-01-01"
ANCHOR_END   = "2016-12-01"

OUT_MONTHLY_HISTORY = "market_price_monthly_history.csv"
OUT_FORECAST = "market_price_forecast_20y_monthly.csv"


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


def to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s)
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())


# -----------------------------
# 1) Load hourly -> monthly average
# -----------------------------
df = pd.read_csv(INPUT_FILE)
print("CSV columns:", list(df.columns))

if DATE_COL is None or DATE_COL not in df.columns:
    DATE_COL = autodetect_datetime_col(df)
    print(f"Using datetime column: {DATE_COL}")

if PRICE_COL not in df.columns:
    raise ValueError(f"PRICE_COL '{PRICE_COL}' not found. Available columns: {list(df.columns)}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
df = df.dropna(subset=[PRICE_COL])

df["date"] = to_month_start(df[DATE_COL])

monthly = (
    df.groupby("date", as_index=False)[PRICE_COL]
    .mean()
    .rename(columns={PRICE_COL: "price_eur_mwh"})
).sort_values("date").reset_index(drop=True)

monthly.to_csv(OUT_MONTHLY_HISTORY, index=False)
print(f"Saved monthly history: {OUT_MONTHLY_HISTORY}")

# Shift for log
monthly["price_shifted"] = monthly["price_eur_mwh"] + PRICE_SHIFT
if (monthly["price_shifted"] <= 0).any():
    raise ValueError("PRICE_SHIFT too small: still non-positive shifted prices. Increase PRICE_SHIFT.")

monthly["y"] = np.log(monthly["price_shifted"])

# -----------------------------
# 2) Fit AR(1) + seasonality on deviations around drifting mean
# -----------------------------
# Anchor level (median shifted price in anchor window)
mask = (monthly["date"] >= pd.Timestamp(ANCHOR_START)) & (monthly["date"] <= pd.Timestamp(ANCHOR_END))
if mask.sum() < 12:
    raise ValueError("Anchor window too short/missing. Adjust ANCHOR_START/ANCHOR_END.")

anchor_date = monthly.loc[mask, "date"].iloc[-1]
anchor_level = 10 #float(monthly.loc[mask, "price_shifted"].median())
mu0 = 10 #float(np.log(anchor_level)) -

# Build mu_t for history
months_from_anchor = (monthly["date"].dt.year - anchor_date.year) * 12 + (monthly["date"].dt.month - anchor_date.month)
monthly["mu"] = mu0 + (months_from_anchor / 12.0) * np.log(1.0 + ANNUAL_DRIFT) + 0.15

# Work with deviations z_t = y_t - mu_t
monthly["z"] = monthly["y"] - monthly["mu"]
monthly["z_lag"] = monthly["z"].shift(1)

# seasonality dummies on month-of-year
monthly["month_num"] = monthly["date"].dt.month.astype(int)
month_dummies = pd.get_dummies(monthly["month_num"], prefix="m", drop_first=True).astype(float)

# Regression: z_t = c + phi*z_{t-1} + season(month) + error
X = pd.concat([monthly["z_lag"], month_dummies], axis=1)
X = sm.add_constant(X)

model_data = pd.concat([monthly["z"], X], axis=1).dropna()

X_mat = model_data[X.columns].astype(float)
y_vec = model_data["z"].astype(float)

model = sm.OLS(y_vec, X_mat).fit()
print(model.summary())

phi = float(model.params["z_lag"])
const = float(model.params["const"])
season_params = {k: float(v) for k, v in model.params.items() if k.startswith("m_")}

print("\n--- Settings used ---")
print(f"PRICE_SHIFT: {PRICE_SHIFT}")
print(f"ANNUAL_DRIFT (g): {ANNUAL_DRIFT:.4f}")
print(f"Anchor window: {ANCHOR_START}..{ANCHOR_END}")
print(f"Anchor date: {anchor_date.date()}, anchor level (shifted): {anchor_level:.2f}")
print(f"Estimated phi: {phi:.4f}")

# -----------------------------
# 3) Forecast 20y monthly (P50, deterministic)
# -----------------------------
last_date = monthly["date"].max()
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                             periods=FORECAST_YEARS * 12, freq="MS")

# start from last observed z_T (deviation)
z_prev = float(monthly["z"].iloc[-1])

z_fore = []
y_fore = []
p_fore = []

for d in future_dates:
    # compute mu_t for this future month
    months_ahead = (d.year - anchor_date.year) * 12 + (d.month - anchor_date.month)
    mu_t = mu0 + (months_ahead / 12.0) * np.log(1.0 + ANNUAL_DRIFT)

    # season effect (month 1 is baseline due to drop_first=True)
    m = d.month
    s_t = season_params.get(f"m_{m}", 0.0)

    # forecast deviation
    z_t = const + phi * z_prev + s_t

    # back to y_t and price
    y_t = mu_t + z_t
    p_t = float(np.exp(y_t) - PRICE_SHIFT)

    z_fore.append(z_t)
    y_fore.append(y_t)
    p_fore.append(p_t)

    z_prev = z_t

out = pd.DataFrame({
    "date": future_dates,
    "price_eur_mwh_forecast": p_fore,
    "log_price_shifted_forecast": y_fore,
    "mu_log_shifted": [mu0 + ((d.year - anchor_date.year) * 12 + (d.month - anchor_date.month)) / 12.0 * np.log(1.0 + ANNUAL_DRIFT)
                       for d in future_dates],
})

out.to_csv(OUT_FORECAST, index=False)
print(f"\nSaved forecast: {OUT_FORECAST}")
