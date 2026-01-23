"""

Merges the Monte Carlo inputs into ONE monthly table for the revenue script.

Inputs
------
1) market_price_mc_30y_monthly_paths.csv
   - keep: date, sim, price_eur_mwh

2) curtailment_mc_30y_quarterly_paths.csv
   - filter: scenario uses only TSO columns
   - keep: quarter, sim, TSO, cr
   - expand quarterly -> monthly by assigning each month the CR of its quarter
   - output columns become wide by TSO:
        cr_tennet, cr_50hertz, cr_amprion, cr_transnetbw
     (same CR repeated for the 3 months of the quarter)

3) capture_factor_mc_30y_monthly_paths.csv
   - filter: scenario == "base"
   - keep: date, sim, capture_factor

4) eeg_strike_mc_30y_monthly_paths.csv
   - keep: date, sim, eeg_strike_eur_mwh

Output
------
mc_inputs_30y_monthly_merged.csv with columns:
    date, sim, price_eur_mwh, capture_factor, eeg_strike_eur_mwh,
    cr_tennet, cr_50hertz, cr_amprion, cr_transnetbw

Notes
-----
- All merges are done on (date, sim) after turning quarterly curtailment into a monthly table.
- If a TSO isn't present in the curtailment file, its column will still exist (NaN).
"""

from __future__ import annotations

import pandas as pd

# =============================
# CONFIG
# =============================
PRICE_FILE = "marketprices/market_price_mc_30y_monthly_paths.csv"
CR_FILE = "curltailment/curtailment_mc_30y_quarterly_paths.csv"
CF_FILE = "capture_factor/capture_factor_mc_30y_monthly_paths.csv"
EEG_FILE = "eeg/eeg_strike_mc_30y_monthly_paths.csv"

# Capture factor scenario to keep
CF_SCENARIO = "base"

# Output
OUT_FILE = "mc_inputs_30y_monthly_merged.csv"

# If you only want one TSO later, keep wide columns anyway (you can select one in revenue)
TSO_COLNAME_MAP = {
    "TenneT": "cr_tennet",
    "50Hertz": "cr_50hertz",
    "Amprion": "cr_amprion",
    "TransnetBW": "cr_transnetbw",
}


# =============================
# LOAD: prices (monthly)
# =============================
prices = pd.read_csv(PRICE_FILE, parse_dates=["date"])
prices["sim"] = pd.to_numeric(prices["sim"], errors="coerce")
prices["price_eur_mwh"] = pd.to_numeric(prices["price_eur_mwh"], errors="coerce")
prices = prices.dropna(subset=["date", "sim", "price_eur_mwh"]).copy()
prices["sim"] = prices["sim"].astype(int)

# Ensure month-start timestamps
prices["date"] = pd.to_datetime(prices["date"]).dt.to_period("M").dt.to_timestamp()

# =============================
# LOAD: capture factor (monthly) -> filter base scenario
# =============================
cf = pd.read_csv(CF_FILE, parse_dates=["date"])
if "scenario" not in cf.columns:
    raise ValueError("CF file must contain 'scenario' column.")
cf = cf.loc[cf["scenario"].astype(str).str.lower() == CF_SCENARIO.lower()].copy()

cf["sim"] = pd.to_numeric(cf["sim"], errors="coerce")
cf["capture_factor"] = pd.to_numeric(cf["capture_factor"], errors="coerce")
cf = cf.dropna(subset=["date", "sim", "capture_factor"]).copy()
cf["sim"] = cf["sim"].astype(int)
cf["date"] = pd.to_datetime(cf["date"]).dt.to_period("M").dt.to_timestamp()

cf = cf[["date", "sim", "capture_factor"]]

# =============================
# LOAD: EEG strike (monthly)
# =============================
eeg = pd.read_csv(EEG_FILE, parse_dates=["date"])
eeg["sim"] = pd.to_numeric(eeg["sim"], errors="coerce")
eeg["eeg_strike_eur_mwh"] = pd.to_numeric(eeg["eeg_strike_eur_mwh"], errors="coerce")
eeg = eeg.dropna(subset=["date", "sim", "eeg_strike_eur_mwh"]).copy()
eeg["sim"] = eeg["sim"].astype(int)
eeg["date"] = pd.to_datetime(eeg["date"]).dt.to_period("M").dt.to_timestamp()

eeg = eeg[["date", "sim", "eeg_strike_eur_mwh"]]

# =============================
# LOAD: Curtailment (quarterly) -> wide by TSO -> monthly via quarter key
# =============================
cr = pd.read_csv(CR_FILE)

required = {"quarter", "sim", "TSO", "cr"}
missing = required - set(cr.columns)
if missing:
    raise ValueError(f"CR file missing columns {missing}. Found: {list(cr.columns)}")

cr["sim"] = pd.to_numeric(cr["sim"], errors="coerce")
cr["cr"] = pd.to_numeric(cr["cr"], errors="coerce")
cr = cr.dropna(subset=["quarter", "sim", "TSO", "cr"]).copy()
cr["sim"] = cr["sim"].astype(int)

# Parse quarter strings like "2026Q1" into Period('Q')
cr["quarter"] = pd.PeriodIndex(cr["quarter"].astype(str), freq="Q")
cr["TSO"] = cr["TSO"].astype(str)

# Pivot to wide: (quarter, sim) rows and one cr column per TSO
cr_wide = (
    cr.pivot_table(index=["quarter", "sim"], columns="TSO", values="cr", aggfunc="mean")
      .reset_index()
)

# Rename TSO columns to stable names (cr_tennet, ...)
for tso, colname in TSO_COLNAME_MAP.items():
    if tso in cr_wide.columns:
        cr_wide = cr_wide.rename(columns={tso: colname})
    else:
        # Ensure column exists even if that TSO missing
        cr_wide[colname] = pd.NA

# Now create a month key for merging: month -> quarter mapping will be done on the monthly table
# We keep the quarter Period plus sim and the TSO CR columns
cr_wide = cr_wide[["quarter", "sim"] + list(TSO_COLNAME_MAP.values())]

# =============================
# BUILD base monthly table and merge quarter-based CR
# =============================
# Add quarter key to prices (monthly) to map each month to its quarter
base = prices.copy()
base["quarter"] = base["date"].dt.to_period("Q")

# Merge CR on (quarter, sim) -> repeats same CR for all 3 months in that quarter
base = base.merge(cr_wide, on=["quarter", "sim"], how="left")

# Drop the helper quarter column after merge
base = base.drop(columns=["quarter"])

# =============================
# Merge CF and EEG on (date, sim)
# =============================
base = base.merge(cf, on=["date", "sim"], how="left")
base = base.merge(eeg, on=["date", "sim"], how="left")

# =============================
# Final ordering + sanity
# =============================
# Optional: sort for readability
base = base.sort_values(["sim", "date"]).reset_index(drop=True)

# Write output
base.to_csv(OUT_FILE, index=False)

print(f"[MERGE] Wrote {len(base):,} rows -> {OUT_FILE}")
print("[MERGE] Columns:", list(base.columns))

# Quick checks (helps catch mismatched horizons / sims)
print("[MERGE] Sims in prices:", prices["sim"].nunique())
print("[MERGE] Sims in merged:", base["sim"].nunique())
print("[MERGE] Date range:", base["date"].min().date(), "..", base["date"].max().date())
