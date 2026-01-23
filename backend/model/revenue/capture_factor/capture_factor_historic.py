"""

Compute historical *monthly* wind capture factors from hourly data.

Definitions
-----------
- Market price (monthly):
    p_market = mean(hourly day-ahead price)

- Wind capture price (monthly, wind-weighted):
    p_capture = sum(price_t * wind_mwh_t) / sum(wind_mwh_t)

- Capture factor:
    CF = p_capture / p_market

Notes / assumptions
-------------------
- Inputs are expected to be hourly and aligned to the same timestamps (timezone included if present).
- We inner-join on timestamps: only hours that exist in BOTH datasets contribute.
- Months are represented as month-start timestamps (e.g., 2020-01-01) for easy plotting/joins.
- Months with zero total wind are kept but CF becomes NaN (avoid divide-by-zero).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================
PRICES_FILE = "../marketprices/prices_hourly_merged.csv"
WIND_FILE = "produced_mwh_wind/wind_onshore_hourly_merged.csv"

# Column names (adjust if your upstream scripts change them)
PRICE_DATE_COL = "datetime"
PRICE_COL = "price_eur_mwh"  # day-ahead price in €/MWh

WIND_DATE_COL = "datetime"
WIND_COL = "wind_onshore_mwh"  # hourly onshore wind generation in MWh

OUTPUT_FILE = "capture_factor_monthly_historical.csv"


# =============================================================================
# Helpers
# =============================================================================
def month_start(ts: pd.Series) -> pd.Series:
    """
    Convert timestamps to a month identifier using month-start timestamps.

    Example:
      2020-01-15 12:00 -> 2020-01-01 00:00
    """
    return ts.dt.to_period("M").dt.to_timestamp()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and minimally clean input datasets.

    - parse datetime columns
    - coerce numeric values
    - drop rows with missing timestamp/value
    """
    prices = pd.read_csv(PRICES_FILE, parse_dates=[PRICE_DATE_COL])
    wind = pd.read_csv(WIND_FILE, parse_dates=[WIND_DATE_COL])

    prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")
    wind[WIND_COL] = pd.to_numeric(wind[WIND_COL], errors="coerce")

    prices = prices.dropna(subset=[PRICE_DATE_COL, PRICE_COL]).copy()
    wind = wind.dropna(subset=[WIND_DATE_COL, WIND_COL]).copy()

    return prices, wind


def build_hourly_panel(prices: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    """
    Align prices and wind to a single hourly panel via an inner join on timestamps.

    Inner join is deliberate:
    - avoids introducing NaNs from hours that exist only in one dataset
    - ensures capture price uses the same hour set as market price for that month
    """
    hourly = prices.merge(
        wind,
        left_on=PRICE_DATE_COL,
        right_on=WIND_DATE_COL,
        how="inner",
        suffixes=("_price", "_wind"),
    )

    # Keep one timestamp column to reduce confusion downstream
    hourly = hourly.rename(columns={PRICE_DATE_COL: "timestamp"})
    hourly = hourly[["timestamp", PRICE_COL, WIND_COL]].copy()

    return hourly


def compute_monthly_capture_factors(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly panel to monthly market price, capture price, and capture factor.
    """
    hourly["month"] = month_start(hourly["timestamp"])

    # 1) Market price: simple (unweighted) mean of hourly day-ahead prices
    monthly_market = (
        hourly.groupby("month", as_index=False)[PRICE_COL]
        .mean()
        .rename(columns={PRICE_COL: "p_market_eur_mwh"})
    )

    # 2) Wind capture price: wind-weighted average
    #    (we keep intermediate sums so results are auditable)
    hourly["price_x_wind"] = hourly[PRICE_COL] * hourly[WIND_COL]

    monthly_wind = (
        hourly.groupby("month", as_index=False)
        .agg(
            wind_mwh=(WIND_COL, "sum"),
            price_x_wind=("price_x_wind", "sum"),
        )
    )

    # Avoid divide-by-zero: months with no wind generation get NaN capture price
    monthly_wind["p_wind_capture_eur_mwh"] = np.where(
        monthly_wind["wind_mwh"] > 0,
        monthly_wind["price_x_wind"] / monthly_wind["wind_mwh"],
        np.nan,
    )

    # 3) Combine + capture factor
    monthly = monthly_market.merge(monthly_wind, on="month", how="inner")

    monthly["capture_factor"] = monthly["p_wind_capture_eur_mwh"] / monthly["p_market_eur_mwh"]

    # Keep a tidy, consistent column order
    monthly = monthly[[
        "month",
        "p_market_eur_mwh",
        "p_wind_capture_eur_mwh",
        "capture_factor",
        "wind_mwh",
    ]]

    return monthly.sort_values("month").reset_index(drop=True)


def main() -> None:
    prices, wind = load_inputs()
    hourly = build_hourly_panel(prices, wind)
    monthly = compute_monthly_capture_factors(hourly)

    monthly.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved historical capture factors to: {OUTPUT_FILE}")

    # Quick sanity stats for a human check during development
    print("\nCapture factor summary:")
    print(monthly["capture_factor"].describe())


if __name__ == "__main__":
    main()
