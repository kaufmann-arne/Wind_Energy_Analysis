"""
EEG Strike Model (Markup-over-Market) — corrected & robust

Goal:
  Build an EEG strike price forecast that tracks your market price scenarios, while the
  markup (strike / market) declines over time toward a floor.

Core idea:
  strike_scen[t] = markup_path[t] * market_price_scen[t]

Markup path (policy assumption):
  markup_t = m_floor + (m0 - m_floor) * exp(-lambda * years_since_anchor), for t >= anchor
  markup_t = m0, for t < anchor

Where:
  - m0 is anchored from recent observed data (or manually set)
  - m_floor is long-run markup floor (e.g. 1.00–1.05)
  - lambda derived from half-life in years

Inputs:
- forecast_market_price_ar1_stochastic_3scen.csv
    must contain a date column (first column), plus scenario p50 columns:
      price_low_p50, price_base_p50, price_high_p50
    and optionally history column:
      price_hist_eur_mwh

- eeg_auction_strike_prices.csv
    columns:
      auction_date (dd.mm.yyyy), strike_eur_mwh

Outputs:
- eeg_markup_and_strike_forecast_3scen.csv
    date, market_used, strike_hist, markup_hist, markup_path,
    strike_low, strike_base, strike_high, strike_used_hist_or_base
"""

import numpy as np
import pandas as pd

# =============================
# CONFIG
# =============================
MARKET_FILE = "forecast_market_price_ar1_stochastic_3scen.csv"
EEG_FILE = "eeg_auction_strike_prices.csv"

# Market column names (change only if your file differs)
HIST_PRICE_COL = "price_hist_eur_mwh"
LOW_P50_COL = "price_low_p50"
BASE_P50_COL = "price_base_p50"
HIGH_P50_COL = "price_high_p50"

# EEG column names
EEG_DATE_COL = "auction_date"
EEG_STRIKE_COL = "strike_eur_mwh"

# Markup model parameters (policy knobs)
ANCHOR_MONTHS = 24          # compute m0 from last 24 months of overlap (or fewer if not available)
M_FLOOR = 1.02              # long-run markup floor, e.g. 1.00–1.05
HALF_LIFE_YEARS = 12        # speed: years for (m - floor) to halve

# Optional: override m0 manually (set to None to estimate from data)
MANUAL_M0 = None            # e.g. 1.15 for "15% over market today"

OUTPUT_FILE = "eeg_markup_and_strike_forecast_3scen.csv"


# =============================
# LOAD MARKET (robust date col)
# =============================
mkt = pd.read_csv(MARKET_FILE, parse_dates=[0])
DATE_COL = mkt.columns[0]  # robust: first column is the date column
mkt = mkt.sort_values(DATE_COL).reset_index(drop=True)

required_mkt_cols = [LOW_P50_COL, BASE_P50_COL, HIGH_P50_COL]
missing = [c for c in required_mkt_cols if c not in mkt.columns]
if missing:
    raise ValueError(f"Missing required market columns: {missing}. Found: {list(mkt.columns)}")

# Create a single "market used" series (for history markup calc):
# use history if available, else fall back to base p50.
mkt["p_market_used"] = np.nan
if HIST_PRICE_COL in mkt.columns:
    mkt["p_market_used"] = mkt[HIST_PRICE_COL]

mkt.loc[mkt["p_market_used"].isna(), "p_market_used"] = mkt.loc[mkt["p_market_used"].isna(), BASE_P50_COL]

# =============================
# LOAD EEG AUCTIONS -> MONTHLY STRIKE SERIES
# =============================
eeg = pd.read_csv(EEG_FILE)
eeg[EEG_DATE_COL] = pd.to_datetime(eeg[EEG_DATE_COL], dayfirst=True, errors="coerce")
eeg[EEG_STRIKE_COL] = pd.to_numeric(eeg[EEG_STRIKE_COL], errors="coerce")
eeg = eeg.dropna(subset=[EEG_DATE_COL, EEG_STRIKE_COL]).copy()

eeg["month"] = eeg[EEG_DATE_COL].dt.to_period("M").dt.to_timestamp()
strike_m = (
    eeg.groupby("month", as_index=False)[EEG_STRIKE_COL]
       .mean()
       .sort_values("month")
       .rename(columns={"month": DATE_COL, EEG_STRIKE_COL: "strike_eur_mwh_hist"})
)

# =============================
# MERGE MARKET + STRIKE
# =============================
df = mkt[[DATE_COL, "p_market_used", LOW_P50_COL, BASE_P50_COL, HIGH_P50_COL]].merge(
    strike_m, on=DATE_COL, how="left"
)

# Historical markup where possible
df["markup_hist"] = df["strike_eur_mwh_hist"] / df["p_market_used"]

# =============================
# CHOOSE m0 (anchor markup)
# =============================
overlap = df.dropna(subset=["markup_hist"]).copy()
if overlap.empty:
    raise ValueError(
        "No overlap between strike history and market series. "
        "Check that dates align monthly and that the market file covers the EEG auction period."
    )

# Use up to ANCHOR_MONTHS of overlap (if you have fewer, it will just use all)
overlap_tail = overlap.tail(min(ANCHOR_MONTHS, len(overlap)))
m0_est = float(overlap_tail["markup_hist"].mean())
m0 = float(MANUAL_M0) if MANUAL_M0 is not None else m0_est

# Anchor date = last month with observed strike (from overlap tail)
anchor_date = pd.Timestamp(overlap_tail[DATE_COL].max())

# =============================
# BUILD MARKUP PATH
# =============================
lam = np.log(2.0) / float(HALF_LIFE_YEARS)

months_from_anchor = (df[DATE_COL].dt.year - anchor_date.year) * 12 + (df[DATE_COL].dt.month - anchor_date.month)
years_from_anchor = months_from_anchor / 12.0

df["markup_path"] = m0
forward = years_from_anchor > 0
df.loc[forward, "markup_path"] = M_FLOOR + (m0 - M_FLOOR) * np.exp(-lam * years_from_anchor[forward])

# =============================
# STRIKE FORECAST PER MARKET SCENARIO
# =============================
df["strike_low"]  = df["markup_path"] * df[LOW_P50_COL]
df["strike_base"] = df["markup_path"] * df[BASE_P50_COL]
df["strike_high"] = df["markup_path"] * df[HIGH_P50_COL]

# For convenience: use historical strike where available, otherwise base forecast
df["strike_used_hist_or_base"] = df["strike_eur_mwh_hist"]
df.loc[df["strike_used_hist_or_base"].isna(), "strike_used_hist_or_base"] = df.loc[
    df["strike_used_hist_or_base"].isna(), "strike_base"
]

# =============================
# SAVE
# =============================
out_cols = [
    DATE_COL,
    "p_market_used",
    "strike_eur_mwh_hist",
    "markup_hist",
    "markup_path",
    "strike_low",
    "strike_base",
    "strike_high",
    "strike_used_hist_or_base",
]
df[out_cols].to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print(f"Date column used: {DATE_COL}")
print(f"Anchor date: {anchor_date.date()}")
print(f"m0 used: {m0:.4f}  (estimated: {m0_est:.4f}, manual override: {MANUAL_M0})")
print(f"Markup floor: {M_FLOOR:.4f}, half-life: {HALF_LIFE_YEARS} years")
