import numpy as np
import pandas as pd

# =============================
# CONFIG
# =============================
MARKET_FILE = "market_price_forecast_20y_monthly.csv"
EEG_FILE = "eeg_auction_strike_prices.csv"

MARKET_DATE_COL = "date"
MARKET_PRICE_COL = "price_eur_mwh_forecast"

EEG_DATE_COL = "auction_date"
EEG_STRIKE_COL = "strike_eur_mwh"

# Policy knobs
M_FLOOR = 1.02            # long-run markup floor (1.00–1.05 typical)
HALF_LIFE_YEARS = 12      # convergence speed

# --- Anchor method (choose one) ---
# A) Manual markup assumption (recommended for your case)
MANUAL_M0 = 1.15          # 15% over market "today" (set None to use method B)

# B) If MANUAL_M0 is None, infer m0 = last_strike / REF_MARKET_PRICE
REF_MARKET_PRICE = 70.0   # €/MWh reference market price for anchor (only used if MANUAL_M0=None)

# Output
OUTPUT_FILE = "eeg_strike_forecast_monthly.csv"


# =============================
# LOAD MARKET (future path)
# =============================
mkt = pd.read_csv(MARKET_FILE, parse_dates=[MARKET_DATE_COL])
mkt = mkt.sort_values(MARKET_DATE_COL).reset_index(drop=True)

# =============================
# LOAD EEG AUCTIONS
# =============================
eeg = pd.read_csv(EEG_FILE)
eeg[EEG_DATE_COL] = pd.to_datetime(eeg[EEG_DATE_COL], dayfirst=True, errors="coerce")
eeg[EEG_STRIKE_COL] = pd.to_numeric(eeg[EEG_STRIKE_COL], errors="coerce")
eeg = eeg.dropna(subset=[EEG_DATE_COL, EEG_STRIKE_COL]).copy()

# monthly auction series
eeg["date"] = eeg[EEG_DATE_COL].dt.to_period("M").dt.to_timestamp()
strike_m = (
    eeg.groupby("date", as_index=False)[EEG_STRIKE_COL]
       .mean()
       .rename(columns={EEG_STRIKE_COL: "strike_hist"})
       .sort_values("date")
       .reset_index(drop=True)
)

# =============================
# ANCHOR m0 (no overlap needed)
# =============================
last_strike_date = pd.Timestamp(strike_m["date"].max())
last_strike = float(strike_m.loc[strike_m["date"] == last_strike_date, "strike_hist"].iloc[0])

if MANUAL_M0 is not None:
    m0 = float(MANUAL_M0)
    anchor_note = f"Manual m0={m0:.3f}"
else:
    if REF_MARKET_PRICE <= 0:
        raise ValueError("REF_MARKET_PRICE must be > 0.")
    m0 = float(last_strike / REF_MARKET_PRICE)
    anchor_note = f"Derived m0=last_strike/REF_MARKET_PRICE={m0:.3f} (ref price {REF_MARKET_PRICE:.2f})"

anchor_date = last_strike_date

# =============================
# BUILD MARKUP PATH ON MARKET DATES
# =============================
lam = np.log(2.0) / float(HALF_LIFE_YEARS)

months_from_anchor = (
    (mkt[MARKET_DATE_COL].dt.year - anchor_date.year) * 12
    + (mkt[MARKET_DATE_COL].dt.month - anchor_date.month)
)
years_from_anchor = months_from_anchor / 12.0

markup_path = np.full(len(mkt), m0, dtype=float)
forward = years_from_anchor > 0
markup_path[forward.values] = M_FLOOR + (m0 - M_FLOOR) * np.exp(-lam * years_from_anchor[forward].values)

mkt["markup_path"] = markup_path

# =============================
# EEG STRIKE FORECAST (2026+)
# =============================
mkt["eeg_strike_eur_mwh"] = mkt["markup_path"] * mkt[MARKET_PRICE_COL]

# =============================
# OPTIONAL: OUTPUT A COMBINED SERIES INCLUDING HIST AUCTIONS
# =============================
# Put historical strikes on the same date axis (auction months),
# and then the forecast from 2026 onward.
hist_out = strike_m.rename(columns={"strike_hist": "eeg_strike_used"}).copy()
hist_out["markup_path"] = np.nan
hist_out["market_price"] = np.nan
hist_out["eeg_strike_eur_mwh"] = np.nan

fcst_out = mkt[[MARKET_DATE_COL, MARKET_PRICE_COL, "markup_path", "eeg_strike_eur_mwh"]].copy()
fcst_out = fcst_out.rename(columns={
    MARKET_DATE_COL: "date",
    MARKET_PRICE_COL: "market_price"
})
fcst_out["eeg_strike_used"] = fcst_out["eeg_strike_eur_mwh"]

out = pd.concat([hist_out[["date","market_price","markup_path","eeg_strike_eur_mwh","eeg_strike_used"]],
                 fcst_out[["date","market_price","markup_path","eeg_strike_eur_mwh","eeg_strike_used"]]],
                ignore_index=True).sort_values("date").reset_index(drop=True)

out.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print(f"Anchor date (last auction month): {anchor_date.date()}  | Last strike: {last_strike:.2f} €/MWh")
print(anchor_note)
print(f"Markup floor: {M_FLOOR:.3f}, half-life: {HALF_LIFE_YEARS} years")
print(f"Forecast starts at: {mkt[MARKET_DATE_COL].min().date()}  (market file start)")
