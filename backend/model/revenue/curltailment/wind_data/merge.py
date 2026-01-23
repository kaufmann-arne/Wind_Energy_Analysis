"""

Purpose
-------
Merge daily SMARD wind-onshore generation CSV files from different TSOs
into a single, clean daily time series.

The output is a *wide* daily table:
    date, TenneT, 50Hertz, Amprion, TransnetBW

Why this script exists
----------------------
SMARD exports are split by TSO and by time period, which leads to:
- multiple overlapping CSVs per TSO
- inconsistent numeric formatting (thousands separators)
- daily values that must be aligned across TSOs

This script:
1) Reads all CSVs per TSO
2) Parses daily dates safely
3) Converts wind generation values to numeric MWh
4) Aggregates to daily totals (defensive even if files overlap)
5) Merges all TSOs into one daily dataset

The result is used downstream for:
- quarterly aggregation
- curtailment proxy modeling
- Monte Carlo simulations
"""

from __future__ import annotations

import os
import glob
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory: keep script and data together for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output file (daily, wide format)
OUT_FILE = os.path.join(BASE_DIR, "wind_onshore_daily_by_tso.csv")

# Column names as used in SMARD exports
DT_COL = "Start date"
WIND_COL = "Wind onshore [MWh] Calculated resolutions"

# Date format used in these SMARD daily exports (e.g. "Jan 1, 2019")
DT_FORMAT = "%b %d, %Y"

# File patterns per TSO
# (Note: SMARD uses "Tennet" not "TenneT" in filenames)
TSO_PATTERNS = {
    "TenneT":      "Tennet*.csv",
    "50Hertz":     "50Hertz*.csv",
    "Amprion":     "Amprion*.csv",
    "TransnetBW":  "TransnetBW*.csv",
}


# =============================================================================
# HELPERS
# =============================================================================
def parse_us_number_series(s: pd.Series) -> pd.Series:
    """
    Convert numeric strings like:
        '16,916.50' -> 16916.50

    Assumptions
    -----------
    - Thousands separator is comma
    - Decimal separator is dot
    - Missing values may appear as empty strings or dashes

    Returns
    -------
    pd.Series of floats (NaN if parsing fails)
    """
    s = s.astype("string").str.strip()

    # Remove thousands separators
    s = s.str.replace(",", "", regex=False)

    # Normalize common placeholders for missing data
    s = s.replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "–": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
        }
    )

    return pd.to_numeric(s, errors="coerce")


def read_one_tso(paths: list[str], tso: str) -> pd.DataFrame:
    """
    Read and merge all CSV files belonging to a single TSO.

    Steps
    -----
    For each file:
    1) Read CSV with correct separator and encoding
    2) Parse daily dates explicitly (fail-safe)
    3) Parse wind-onshore MWh values
    4) Aggregate to daily totals (safe even if data is already daily)

    After processing all files:
    - concatenate
    - sort by date
    - deduplicate (keep last in case of overlaps)

    Returns
    -------
    DataFrame with columns:
        date, <TSO>
    """
    parts = []

    for fp in sorted(paths):
        df = pd.read_csv(
            fp,
            sep=";",
            encoding="utf-8-sig",
            dtype="string",
            engine="python",
        )

        # Fail early if expected columns are missing
        if DT_COL not in df.columns:
            raise ValueError(f"[{tso}] Missing '{DT_COL}' in {fp}")
        if WIND_COL not in df.columns:
            raise ValueError(f"[{tso}] Missing '{WIND_COL}' in {fp}")

        # Parse date column using explicit daily format
        dt = pd.to_datetime(df[DT_COL], format=DT_FORMAT, errors="coerce")

        # Parse numeric wind generation values
        wind = parse_us_number_series(df[WIND_COL])

        tmp = pd.DataFrame({"date": dt, tso: wind})

        before = len(tmp)

        # Drop rows with invalid dates
        tmp = tmp.dropna(subset=["date"])

        # Normalize date to midnight (ensures consistent daily key)
        tmp["date"] = tmp["date"].dt.normalize()

        # Drop rows where wind could not be parsed
        tmp = tmp.dropna(subset=[tso])

        # Aggregate by day (important if overlapping or duplicated rows exist)
        tmp = tmp.groupby("date", as_index=False)[tso].sum()

        print(
            f"[{tso}] {os.path.basename(fp)}: "
            f"raw_rows={before:,} -> daily_rows={len(tmp):,}"
        )

        parts.append(tmp)

    # Combine all files for this TSO
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", tso])

    # Stable sort + deduplication ensures deterministic results
    out = (
        out.sort_values("date", kind="mergesort")
           .drop_duplicates(subset=["date"], keep="last")
           .reset_index(drop=True)
    )

    # Safety check: never silently return an empty dataset
    if len(out) == 0:
        raise RuntimeError(
            f"[{tso}] Output is empty. Date parsing likely failed "
            f"(expected format '{DT_FORMAT}')."
        )

    return out


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """
    Script entry point.

    For each TSO:
    - locate matching CSV files
    - read and clean them into a daily series

    Then:
    - merge all TSOs on date (outer join)
    - write final daily dataset to disk
    """
    series_list = []

    for tso, pattern in TSO_PATTERNS.items():
        paths = glob.glob(os.path.join(BASE_DIR, pattern))
        print(f"\nSearching {tso} with pattern '{pattern}': matched {len(paths)} files")

        if not paths:
            raise FileNotFoundError(f"No files found for {tso} using pattern: {pattern}")

        series_list.append(read_one_tso(paths, tso))

    # Merge all TSOs on the date column
    merged = series_list[0]
    for df in series_list[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Basic diagnostics (useful when adding new data)
    print("\nRows:", len(merged))
    print("Date range:", merged["date"].min().date(), "→", merged["date"].max().date())
    print("\nMissing values per column:")
    print(merged.isna().sum())

    if len(merged) == 0:
        raise RuntimeError("Merged output is empty. Aborting to avoid downstream errors.")

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
