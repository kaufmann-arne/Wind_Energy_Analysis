"""

Aggregate SMARD-style TSO grid load CSVs into one *monthly* wide table.

Expected output columns
-----------------------
date,
  load_mwh_tennet,
  load_mwh_amprion,
  load_mwh_50hertz,
  load_mwh_transnetbw

Assumptions / interpretation
----------------------------
- Input files are semicolon-separated (";") SMARD exports.
- "Start date" contains timestamps or dates that can be mapped to a month.
- LOAD_COL contains numeric strings like "9,399,413.00" (commas as thousands separators).
- Within each file we aggregate by month using sum().
  This is safe for:
    * monthly files (sum is effectively identity if there’s one row per month),
    * daily/hourly files (sum produces monthly totals).
  If your input already represents monthly *averages*, sum would be wrong — switch to mean().

Usage
-----
Put this script in the folder with your CSVs and run:
  python aggregate_tso_load.py
"""

from __future__ import annotations

import glob
import os

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Map from a keyword found in the filename -> output column suffix.
# Keep suffixes stable so downstream code can rely on column names.
TSO_PATTERNS = {
    "TenneT": "tennet",
    "Amprion": "amprion",
    "50Hertz": "50hertz",
    "TransnetBW": "transnetbw",
    "TransnetBw": "transnetbw",
    "Transnet": "transnetbw",
}

# Column to extract from each CSV
LOAD_COL = "grid load [MWh] Calculated resolutions"

# Which files to include (script assumes it sits next to the exports)
FILE_GLOB = "*.csv"

# Output filename
OUTPUT_FILE = "tso_grid_load_monthly_wide.csv"

# Date column in SMARD exports
DT_COL = "Start date"

# Common SMARD date format for monthly-ish exports (e.g., "Jan 1, 2015")
DATE_FORMAT_FALLBACK = "%b %d, %Y"


# =============================================================================
# Helpers
# =============================================================================
def detect_tso_from_filename(fname: str) -> str:
    """
    Infer the TSO from the filename based on simple substring matching.

    Implementation detail:
    - We sort keys by length (descending) so specific names match before generic ones
      (e.g., "TransnetBW" before "Transnet").
    """
    base = os.path.basename(fname).lower()

    for key in sorted(TSO_PATTERNS.keys(), key=len, reverse=True):
        if key.lower() in base:
            return TSO_PATTERNS[key]

    raise ValueError(f"Could not detect TSO from filename: {fname}")


def parse_number_series(s: pd.Series) -> pd.Series:
    """
    Parse numeric strings like "9,399,413.00" into floats.

    We remove commas as thousands separators and use pandas' numeric conversion.
    Any unparseable values become NaN and are handled later.
    """
    s = s.astype("string").str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def month_start_from_start_date(s: pd.Series) -> pd.Series:
    """
    Convert a date/timestamp series into month-start timestamps.

    We try an explicit format first for determinism, then fall back to pandas parsing.
    """
    dt = pd.to_datetime(s, format=DATE_FORMAT_FALLBACK, errors="coerce")

    if dt.isna().any():
        # Some files may include time-of-day; fall back to more flexible parsing.
        dt2 = pd.to_datetime(s, errors="coerce")
        dt = dt.fillna(dt2)

    if dt.isna().any():
        raise ValueError("Some Start date values could not be parsed. Check date format / locale.")

    return dt.dt.to_period("M").dt.to_timestamp()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    files = sorted(glob.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched FILE_GLOB='{FILE_GLOB}' in {os.getcwd()}")

    monthly_parts: list[pd.DataFrame] = []

    for f in files:
        df = pd.read_csv(f, sep=";", engine="python")

        # Fail early if expected columns are missing
        if DT_COL not in df.columns:
            raise ValueError(f"'{DT_COL}' column not found in {f}. Columns: {list(df.columns)}")
        if LOAD_COL not in df.columns:
            raise ValueError(f"'{LOAD_COL}' not found in {f}. Columns: {list(df.columns)}")

        tso = detect_tso_from_filename(f)

        tmp = df[[DT_COL, LOAD_COL]].copy()
        tmp["date"] = month_start_from_start_date(tmp[DT_COL])
        tmp["value"] = parse_number_series(tmp[LOAD_COL])

        # Drop rows where the value could not be parsed (keeps output clean and auditable)
        tmp = tmp.dropna(subset=["date", "value"])

        # Aggregate by month. Using sum() is deliberate (see module docstring).
        tmp = tmp.groupby("date", as_index=False)["value"].sum()
        tmp["tso"] = tso

        monthly_parts.append(tmp)
        print(f"Loaded {os.path.basename(f)} -> TSO={tso}, months={len(tmp)}")

    long = pd.concat(monthly_parts, ignore_index=True)

    # Pivot to wide format:
    # index: date
    # columns: tso suffix
    # values: monthly total load
    wide = (
        long.pivot_table(index="date", columns="tso", values="value", aggfunc="sum")
        .reset_index()
    )

    # Make column names explicit and consistent
    wide = wide.rename(columns={c: f"load_mwh_{c}" for c in wide.columns if c != "date"})

    wide = wide.sort_values("date").reset_index(drop=True)

    wide.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print("Columns:", list(wide.columns))


if __name__ == "__main__":
    main()
