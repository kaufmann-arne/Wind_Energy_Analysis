"""

Merge SMARD "Wind onshore" hourly exports into a single clean time series.

What this script solves
-----------------------
- SMARD exports are often split across multiple CSVs and can overlap in time.
- Numeric values can appear in mixed EU/US formatting and include symbols/spaces.
- We want one hourly value per timestamp:
    * sorted
    * deduplicated (later files overwrite earlier values)
    * numeric (float), with invalid rows removed

Output
------
CSV with columns:
- datetime (timestamp)
- wind_onshore_mwh (float)
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Input files (SMARD exports). Keep glob broad so dropping in new exports "just works".
WIND_GLOB = "wh*.csv"

# Output file used by downstream processing / dashboarding
OUT_WIND = "wind_onshore_hourly_merged.csv"

# SMARD column names
DT_COL = "Start date"
WIND_COL = "Wind onshore [MWh] Calculated resolutions"

# SMARD datetime string format (e.g., "Jan 01, 2020 01:00 PM")
DT_FORMAT = "%b %d, %Y %I:%M %p"


# =============================================================================
# Helpers
# =============================================================================
def parse_dt(series: pd.Series) -> pd.Series:
    """
    Parse SMARD timestamps into pandas datetimes.
    Invalid parses become NaT and are filtered out later.
    """
    return pd.to_datetime(series, format=DT_FORMAT, errors="coerce")


def to_float(series: pd.Series) -> pd.Series:
    """
    Convert messy numeric strings into floats.

    SMARD exports can contain:
    - EU format: "1.234,56"
    - US format: "1,234.56"
    - non-breaking spaces and extra units/symbols
    - missing tokens like "-", "–", "", "nan"

    Strategy:
    - Clean up whitespace and missing tokens
    - Strip everything except digits, comma, dot, minus
    - If both comma and dot exist, infer decimal separator by which appears last
    - Convert using pandas; failures -> NaN
    """
    # pandas "string" dtype is safer than astype(str):
    # it preserves missing values as <NA> instead of turning them into the literal "nan".
    s = series.astype("string")

    # Normalize whitespace (SMARD sometimes uses NBSP variants)
    s = (
        s.str.replace("\u00a0", " ", regex=False)
         .str.replace("\u202f", " ", regex=False)
         .str.strip()
    )

    # Normalize common missing tokens
    s = s.replace({"": pd.NA, "-": pd.NA, "–": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Remove everything except digits, separators, and minus sign
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)

    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains(r"\.", na=False)
    both = has_comma & has_dot

    # Both separators present -> decide by which separator appears last
    eu_style = both & (s.str.rfind(",") > s.str.rfind("."))
    s = s.where(
        ~eu_style,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )

    us_style = both & ~eu_style
    s = s.where(
        ~us_style,
        s.str.replace(",", "", regex=False),
    )

    # Only comma -> treat comma as decimal separator
    only_comma = has_comma & ~has_dot
    s = s.where(
        ~only_comma,
        s.str.replace(",", ".", regex=False),
    )

    return pd.to_numeric(s, errors="coerce")


def read_smard_csv_minimal(path: str) -> pd.DataFrame:
    """
    Read only the columns we care about (timestamp + wind value).

    This keeps memory use predictable even if SMARD adds extra columns over time.
    """
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        usecols=[DT_COL, WIND_COL],
    )


def merge_and_dedupe(parts: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    """
    Merge multiple file chunks into a single time series.

    Deduplication policy:
    - Sort by datetime (stable sort)
    - If multiple values exist for the same timestamp, keep the last one
      (later file overwrites earlier data)
    """
    if not parts:
        return pd.DataFrame(columns=["datetime", value_col])

    df = pd.concat(parts, ignore_index=True)

    # Drop unusable rows before sorting/deduping
    df = df.dropna(subset=["datetime", value_col])

    # Stable sort makes "keep last" deterministic with overlapping exports
    df = df.sort_values("datetime", kind="mergesort")

    df = df.drop_duplicates(subset=["datetime"], keep="last")

    return df.reset_index(drop=True)


# =============================================================================
# Main build logic
# =============================================================================
def build_wind() -> pd.DataFrame:
    """
    Build a clean hourly onshore wind time series from all matching CSV files.

    Per file:
    - Parse timestamps
    - Parse wind values to float
    - Drop rows where either field is missing

    Final:
    - One row per timestamp
    - Sorted and deduplicated across files
    """
    files = sorted(glob.glob(WIND_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {WIND_GLOB}")

    parts: list[pd.DataFrame] = []

    for fp in files:
        raw = read_smard_csv_minimal(fp)

        # Parse timestamps first so we can drop invalid rows early
        raw["datetime"] = parse_dt(raw[DT_COL])
        raw = raw[raw["datetime"].notna()].copy()

        out = pd.DataFrame(
            {
                "datetime": raw["datetime"],
                "wind_onshore_mwh": to_float(raw[WIND_COL]),
            }
        )

        # Drop rows where parsing failed
        out = out.dropna(subset=["wind_onshore_mwh"])

        print(f"[WIND ] {fp}: kept {len(out):,} rows")
        parts.append(out)

    return merge_and_dedupe(parts, "wind_onshore_mwh")


def main() -> None:
    """CLI entry point: build merged wind series and write it to OUT_WIND."""
    wind = build_wind()
    wind.to_csv(OUT_WIND, index=False)
    print(f"\nWrote {len(wind):,} rows -> {OUT_WIND}")


if __name__ == "__main__":
    main()
