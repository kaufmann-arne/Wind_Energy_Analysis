"""
Aggregate monthly SMARD-style TSO files (e.g., TenneT1518.csv + TenneT1925.csv, etc.)
into ONE wide CSV:

Output columns:
  date, load_mwh_tennet, load_mwh_amprion, load_mwh_50hertz, load_mwh_transnetbw

Assumptions:
- Files are semicolon-separated (;)
- Decimal is dot, thousands are commas (e.g. 9,399,413.00)
- Dates look like "Jan 1, 2015" (English month names)
- The load column to use is: "grid load [MWh] Calculated resolutions"
  (change LOAD_COL below if you want a different one)

Put this script in the folder with your TSO CSVs and run:
  python aggregate_tso_load.py
"""

import glob
import os
import re
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
# Map from "TSO name in filename" -> output column suffix
TSO_PATTERNS = {
    "TenneT": "tennet",
    "Amprion": "amprion",
    "50Hertz": "50hertz",
    "TransnetBW": "transnetbw",
    "TransnetBw": "transnetbw",
    "Transnet": "transnetbw",
}

LOAD_COL = "grid load [MWh] Calculated resolutions"

# If your filenames differ, adjust this pattern
FILE_GLOB = "*.csv"

OUTPUT_FILE = "tso_grid_load_monthly_wide.csv"

# -----------------------------
def detect_tso_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    for k, v in TSO_PATTERNS.items():
        if k.lower() in base.lower():
            return v
    raise ValueError(f"Could not detect TSO from filename: {fname}")

def parse_number_series(s: pd.Series) -> pd.Series:
    """
    Convert strings like '9,399,413.00' to float 9399413.00
    """
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.strip()
         .replace({"": None, "nan": None})
         .astype(float)
    )

def month_start_from_start_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        # try with English month names explicitly (usually not needed on Windows, but safe)
        dt = pd.to_datetime(s, format="%b %d, %Y", errors="coerce")
    if dt.isna().any():
        raise ValueError("Some Start date values could not be parsed. Check date format / locale.")
    return dt.dt.to_period("M").dt.to_timestamp()

# -----------------------------
def main():
    files = sorted(glob.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched FILE_GLOB='{FILE_GLOB}' in {os.getcwd()}")

    all_series = []  # list of (date, value, tso)

    for f in files:
        # Read with semicolon separator
        df = pd.read_csv(f, sep=";", engine="python")

        # Basic column checks
        if "Start date" not in df.columns:
            raise ValueError(f"'Start date' column not found in {f}. Columns: {list(df.columns)}")
        if LOAD_COL not in df.columns:
            raise ValueError(f"'{LOAD_COL}' not found in {f}. Columns: {list(df.columns)}")

        tso = detect_tso_from_filename(f)

        tmp = df[["Start date", LOAD_COL]].copy()
        tmp["date"] = month_start_from_start_date(tmp["Start date"])
        tmp["value"] = parse_number_series(tmp[LOAD_COL])

        tmp = tmp[["date", "value"]].groupby("date", as_index=False)["value"].sum()
        tmp["tso"] = tso
        all_series.append(tmp)

        print(f"Loaded {f} -> TSO={tso}, rows={len(tmp)}")

    long = pd.concat(all_series, ignore_index=True)

    # Pivot to wide
    wide = long.pivot_table(index="date", columns="tso", values="value", aggfunc="sum").reset_index()

    # Rename columns to explicit names
    wide = wide.rename(columns={c: f"load_mwh_{c}" for c in wide.columns if c != "date"})

    # Sort by date
    wide = wide.sort_values("date").reset_index(drop=True)

    wide.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print("Columns:", list(wide.columns))

if __name__ == "__main__":
    main()
