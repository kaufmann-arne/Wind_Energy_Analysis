import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
PRICES_FILE = "../day_ahead_market_prices/prices_hourly_merged.csv"
WIND_FILE   = "../produced_mwh_wind/wind_onshore_hourly_merged.csv"

# Column names – ADJUST if needed
PRICE_DATE_COL = "datetime"
PRICE_COL      = "price_eur_mwh"        # day-ahead price €/MWh

WIND_DATE_COL  = "datetime"
WIND_COL       = "wind_onshore_mwh"     # hourly wind generation in MWh

OUTPUT_FILE = "capture_factor_monthly_historical.csv"

# -----------------------------
# 1. Load data
# -----------------------------
prices = pd.read_csv(PRICES_FILE, parse_dates=[PRICE_DATE_COL])
wind   = pd.read_csv(WIND_FILE,   parse_dates=[WIND_DATE_COL])

# ensure numeric
prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")
wind[WIND_COL]    = pd.to_numeric(wind[WIND_COL], errors="coerce")

prices = prices.dropna(subset=[PRICE_DATE_COL, PRICE_COL])
wind   = wind.dropna(subset=[WIND_DATE_COL, WIND_COL])

# -----------------------------
# 2. Merge hourly price & wind
# -----------------------------
hourly = prices.merge(
    wind,
    left_on=PRICE_DATE_COL,
    right_on=WIND_DATE_COL,
    how="inner",
    suffixes=("_price", "_wind")
)

# keep one timestamp column only
hourly["timestamp"] = hourly[PRICE_DATE_COL]
hourly = hourly[["timestamp", PRICE_COL, WIND_COL]]

# -----------------------------
# 3. Add month identifier
# -----------------------------
hourly["month"] = hourly["timestamp"].dt.to_period("M").dt.to_timestamp()

# -----------------------------
# 4. Monthly aggregation
# -----------------------------
# Market price: simple average
monthly_market = (
    hourly.groupby("month")[PRICE_COL]
    .mean()
    .rename("p_market_eur_mwh")
)

# Wind capture price: wind-weighted average
hourly["price_x_wind"] = hourly[PRICE_COL] * hourly[WIND_COL]

monthly_wind = (
    hourly.groupby("month")
    .agg(
        wind_mwh=(WIND_COL, "sum"),
        price_x_wind=("price_x_wind", "sum")
    )
)

monthly_wind["p_wind_capture_eur_mwh"] = (
    monthly_wind["price_x_wind"] / monthly_wind["wind_mwh"]
)

# -----------------------------
# 5. Combine & compute CF
# -----------------------------
monthly = pd.concat([monthly_market, monthly_wind], axis=1).reset_index()

monthly["capture_factor"] = (
    monthly["p_wind_capture_eur_mwh"] / monthly["p_market_eur_mwh"]
)

# -----------------------------
# 6. Clean up & save
# -----------------------------
monthly = monthly.sort_values("month").reset_index(drop=True)

monthly.to_csv(OUTPUT_FILE, index=False)
print(f"Saved historical capture factors to: {OUTPUT_FILE}")

# Optional: quick sanity stats
print("\nCapture factor summary:")
print(monthly["capture_factor"].describe())
