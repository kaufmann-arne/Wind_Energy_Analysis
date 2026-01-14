import os
import re
import glob
import pandas as pd

# -----------------------------
# Config
# -----------------------------
INPUT_DIR = "."   # <-- change to your folder
OUTPUT_FILE = "netztransparenz_wind_onshore_monthly.csv"

# Accept both old and new naming variants
WIND_ROW_REGEX = r"^MW\s+Wind\s+(Onshore|an\s+Land)$"

# German month abbreviations
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mär": 3, "Mrz": 3, "Apr": 4, "Mai": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Okt": 10, "Nov": 11, "Dez": 12
}

def parse_float_ct_per_kwh(x):
    """Convert ' 4,645' -> 4.645"""
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s == "-":
        return float("nan")
    return float(s.replace(" ", "").replace(",", "."))

def extract_from_file(path):
    raw = pd.read_csv(
        path,
        sep=";",
        header=None,
        dtype=str,
        encoding="utf-8",
        engine="python"
    ).dropna(how="all")

    # -------------------------------------------------
    # Parse header row → dates
    # -------------------------------------------------
    header = raw.iloc[0].tolist()

    dates = []
    for cell in header[1:13]:
        token = str(cell).strip()
        m = re.match(r"([A-Za-zÄÖÜäöü]{3})\s+(\d{4})", token)
        if not m:
            raise ValueError(f"Cannot parse date from header cell '{cell}' in {path}")

        month_abbr, year = m.group(1), int(m.group(2))
        month = MONTH_MAP[month_abbr]
        dates.append(pd.Timestamp(year=year, month=month, day=1))

    # -------------------------------------------------
    # Find wind onshore row
    # -------------------------------------------------
    raw[0] = raw[0].fillna("").astype(str).str.strip()

    matches = raw[raw[0].str.match(WIND_ROW_REGEX, case=False)]
    if matches.empty:
        raise ValueError(f"Wind onshore row not found in {path}")

    row = matches.iloc[0]
    values_ct = [parse_float_ct_per_kwh(v) for v in row.iloc[1:13]]
    values_eur_mwh = [v * 10 for v in values_ct]  # ct/kWh → €/MWh

    return pd.DataFrame({
        "date": dates,
        "wind_onshore_eur_mwh": values_eur_mwh
    })

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "wp*.csv")))
    if not files:
        raise FileNotFoundError("No wp*.csv files found")

    frames = [extract_from_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Deduplicate and sort
    df = (
        df.sort_values("date")
          .drop_duplicates(subset=["date"], keep="last")
          .reset_index(drop=True)
    )

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} with {len(df)} rows")

if __name__ == "__main__":
    main()