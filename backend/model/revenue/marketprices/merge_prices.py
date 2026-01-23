"""
prices_merge.py

Merge historical hourly electricity prices exported from SMARD (Germany) into one clean time series.

Why this script exists
----------------------
SMARD exports are convenient but not consistent across time and exports:

- Column names changed around 2018-10-01 (old DE/AT/LU label vs new Germany/Luxembourg label).
- Price values are stored as strings and show up in mixed number formats (EU/US), sometimes with
  symbols/spaces, and sometimes missing-value placeholders.
- Multiple exports can overlap in time. We want a single value per timestamp, and we assume later exports
  should overwrite earlier ones for the same hour.

Output contract
---------------
- One row per hourly timestamp (as provided by SMARD).
- Columns: ["datetime", "price_eur_mwh"]
- Sorted ascending by datetime.
- Deduplicated; if the same timestamp appears multiple times, the last occurrence wins.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# SMARD exports (semicolon-separated CSV). Pattern is intentionally broad so new exports are picked up automatically.
PRICE_GLOB = "data/eh*.csv"

# Output file for downstream scripts (forecasting, dashboards, etc.)
OUT_PRICE = "prices_hourly_merged.csv"

# Column containing timestamps (SMARD label)
DT_COL = "Start date"

# Price column names used by SMARD at different points in time.
# Note: Some exports contain only one of these; some contain both.
PRICE_COL_OLD = "DE/AT/LU [€/MWh] Calculated resolutions"
PRICE_COL_NEW = "Germany/Luxembourg [€/MWh] Calculated resolutions"

# The date used as a split point when both columns are present.
# We treat timestamps < switch date as "old naming", >= switch date as "new naming".
PRICE_SWITCH_DATE = pd.Timestamp("2018-10-01")

# Datetime format used in SMARD exports (example: "Jan 01, 2018 01:00 PM")
DT_FORMAT = "%b %d, %Y %I:%M %p"


# =============================================================================
# Helpers
# =============================================================================
def to_float(series: pd.Series) -> pd.Series:
    """
    Parse SMARD price strings into floats (EUR/MWh).

    SMARD exports tend to be messy because:
    - Mixed formatting: "1.234,56" (EU) vs "1,234.56" (US)
    - Non-breaking spaces and various dash characters
    - Currency symbols / units occasionally show up
    - Missing values can appear as "", "-", "–", "nan", etc.

    Parsing strategy (defensive, but fast enough for hourly data):
    1) Normalize whitespace + common missing tokens
    2) Strip to a minimal set of valid characters: digits, minus, comma, dot
    3) Decide how to treat separators:
       - If both comma and dot exist: whichever appears last is treated as decimal separator
       - If only comma exists: treat comma as decimal separator
       - If only dot exists: treat dot as decimal separator (default numeric parsing)
    4) Convert with pandas; invalid parses become NaN
    """
    s = series.astype("string")

    # Normalize whitespace; SMARD sometimes uses NBSP variants that look like normal spaces.
    s = (
        s.str.replace("\u00a0", " ", regex=False)   # NBSP
         .str.replace("\u202f", " ", regex=False)   # narrow NBSP
         .str.strip()
    )

    # Normalize common "missing" tokens early so they become NaN later.
    s = s.replace(
        {"": pd.NA, "-": pd.NA, "–": pd.NA, "nan": pd.NA, "None": pd.NA}
    )

    # Keep only the characters we need for numeric parsing.
    # This drops currency symbols, units, spaces, etc.
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)

    # Separator detection
    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains(r"\.", na=False)
    both = has_comma & has_dot

    # If both separators exist, infer style by the last separator:
    # - EU: "1.234,56" -> decimal is comma -> remove dots (thousands) and swap comma->dot
    eu_style = both & (s.str.rfind(",") > s.str.rfind("."))
    s = s.where(
        ~eu_style,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )

    # - US: "1,234.56" -> decimal is dot -> remove commas (thousands)
    us_style = both & ~eu_style
    s = s.where(
        ~us_style,
        s.str.replace(",", "", regex=False),
    )

    # If only comma exists, treat it as the decimal separator.
    only_comma = has_comma & ~has_dot
    s = s.where(
        ~only_comma,
        s.str.replace(",", ".", regex=False),
    )

    return pd.to_numeric(s, errors="coerce")


def read_smard_csv_minimal(path: str) -> pd.DataFrame:
    """
    Read only the timestamp column and the relevant price column(s) from a SMARD CSV.

    Practical reason:
    - Exports can be large; reading only needed columns reduces memory and speeds up IO.
    - Different exports contain different price column names, so we detect them from the header first.

    Returns a DataFrame with:
      - datetime: parsed pd.Timestamp (invalid parses are NaT)
      - one or both raw price string columns (depending on file contents)
    """
    # Read only the header so we can decide which columns to load.
    header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0)

    cols = [DT_COL]
    if PRICE_COL_OLD in header.columns:
        cols.append(PRICE_COL_OLD)
    if PRICE_COL_NEW in header.columns:
        cols.append(PRICE_COL_NEW)

    if len(cols) == 1:
        raise ValueError(f"{path} contains no recognized price columns")

    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        usecols=cols,
        # Parse datetime while reading to avoid extra passes/copies later.
        converters={DT_COL: lambda x: pd.to_datetime(x, format=DT_FORMAT, errors="coerce")},
    )

    return df.rename(columns={DT_COL: "datetime"})


def merge_and_dedupe(parts: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    """
    Merge multiple partial hourly DataFrames into a single series.

    Deduplication rule ("later wins"):
    - We concatenate parts in the order we processed files.
    - We sort by datetime (stable mergesort).
    - If the same timestamp appears multiple times, we keep the *last* row.

    This is a simple, explicit overwrite strategy for overlapping exports.
    """
    if not parts:
        return pd.DataFrame(columns=["datetime", value_col])

    df = pd.concat(parts, ignore_index=True)

    # If either timestamp or price is missing, the row is useless for downstream processing.
    df = df.dropna(subset=["datetime", value_col])

    # Stable sort so "keep last" is deterministic after concatenation.
    df = df.sort_values("datetime", kind="mergesort")

    # One row per timestamp; last occurrence wins (newer file overwrites older one).
    df = df.drop_duplicates(subset=["datetime"], keep="last")

    return df.reset_index(drop=True)


# =============================================================================
# Main build logic
# =============================================================================
def build_prices() -> pd.DataFrame:
    """
    Build a clean hourly series from all matching SMARD export CSVs.

    Per file:
    - Load minimal set of columns
    - Drop invalid timestamps early
    - Split into "old column" and "new column" time ranges (when present)
    - Parse prices to float and drop rows that fail parsing

    Final result:
    - Columns: datetime, price_eur_mwh
    - Sorted, NaN-free, deduplicated
    """
    files = sorted(glob.glob(PRICE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {PRICE_GLOB}")

    parts: list[pd.DataFrame] = []

    for fp in files:
        raw = read_smard_csv_minimal(fp)

        # Invalid timestamp rows are not useful and also complicate comparisons to switch date.
        raw = raw[raw["datetime"].notna()].copy()

        out_frames: list[pd.DataFrame] = []

        # Old naming period: timestamps strictly before switch date
        if PRICE_COL_OLD in raw.columns:
            mask_old = raw["datetime"] < PRICE_SWITCH_DATE
            if mask_old.any():
                out_frames.append(
                    pd.DataFrame(
                        {
                            "datetime": raw.loc[mask_old, "datetime"],
                            "price_eur_mwh": to_float(raw.loc[mask_old, PRICE_COL_OLD]),
                        }
                    )
                )

        # New naming period: timestamps on/after switch date
        if PRICE_COL_NEW in raw.columns:
            mask_new = raw["datetime"] >= PRICE_SWITCH_DATE
            if mask_new.any():
                out_frames.append(
                    pd.DataFrame(
                        {
                            "datetime": raw.loc[mask_new, "datetime"],
                            "price_eur_mwh": to_float(raw.loc[mask_new, PRICE_COL_NEW]),
                        }
                    )
                )

        if not out_frames:
            # Some exports might include neither column or have no rows in relevant ranges.
            print(f"[PRICE] {fp}: no usable rows")
            continue

        out = pd.concat(out_frames, ignore_index=True)

        # Anything we couldn't parse to float becomes NaN; drop those hours.
        out = out.dropna(subset=["price_eur_mwh"])

        print(f"[PRICE] {fp}: kept {len(out):,} rows")
        parts.append(out)

    return merge_and_dedupe(parts, "price_eur_mwh")


def main() -> None:
    """CLI entry point: build the merged series and write it to OUT_PRICE."""
    prices = build_prices()
    prices.to_csv(OUT_PRICE, index=False)
    print(f"\nWrote {len(prices):,} rows -> {OUT_PRICE}")


if __name__ == "__main__":
    main()
