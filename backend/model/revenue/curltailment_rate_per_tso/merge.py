import os
import glob
import re
import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# Put the script in the same folder as your CSVs (recommended).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Which column to extract from each file:
VALUE_COL = "Wind onshore [MWh] Calculated resolutions"

# Output file
OUT_FILE = os.path.join(BASE_DIR, "wind_onshore_daily_by_tso.csv")

# Expected TSOs and filename patterns.
# Adjust these if your filenames differ.
TSO_PATTERNS = {
    "TenneT":      "Tennet*.csv",       # note: your file uses 'Tennet' not 'TenneT'
    "50Hertz":     "50Hertz*.csv",
    "Amprion":     "Amprion*.csv",
    "TransnetBW":  "TransnetBW*.csv",
}

# If your filenames are slightly different, add more patterns or rename keys.

# -------------------------------------------------
def parse_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert strings like '29,339.75' (US format with thousands comma)
    or '29339.75' into float.
    """
    s = s.astype(str).str.strip()
    # remove thousands separators (commas) but keep decimal dot
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def read_one_tso(files, tso_name: str) -> pd.DataFrame:
    """
    Read all files for one TSO, extract Start date + VALUE_COL,
    return a daily series with columns: date, <tso_name>.
    """
    parts = []
    for f in sorted(files):
        df = pd.read_csv(f, sep=";", dtype=str, encoding="utf-8", engine="python")

        # Basic column checks
        if "Start date" not in df.columns:
            raise ValueError(f"'Start date' column not found in {f}")
        if VALUE_COL not in df.columns:
            raise ValueError(f"'{VALUE_COL}' not found in {f}. Available columns: {list(df.columns)}")

        tmp = df[["Start date", VALUE_COL]].copy()
        tmp = tmp.rename(columns={"Start date": "date", VALUE_COL: tso_name})

        # Parse date (examples: 'Jan 1, 2015')
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")

        # Parse numeric
        tmp[tso_name] = parse_numeric_series(tmp[tso_name])

        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True)

    # Drop bad dates, sort, dedupe (keep last in case of overlaps)
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # Keep only date part (daily frequency)
    out["date"] = out["date"].dt.normalize()

    return out

def main():
    series_list = []

    # Find and read each TSO
    for tso, pattern in TSO_PATTERNS.items():
        paths = glob.glob(os.path.join(BASE_DIR, pattern))
        if not paths:
            raise FileNotFoundError(f"No files found for {tso} using pattern: {pattern}")

        df_tso = read_one_tso(paths, tso)
        series_list.append(df_tso)

    # Merge all TSOs on date (outer join to keep full coverage)
    merged = series_list[0]
    for df in series_list[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Optional: check missingness
    print("Rows:", len(merged))
    print("Date range:", merged["date"].min().date(), "→", merged["date"].max().date())
    print("\nMissing values per column:")
    print(merged.isna().sum())

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")

if __name__ == "__main__":
    main()
