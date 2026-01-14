import pandas as pd
import numpy as np
import glob
import os

# -----------------------------
# CONFIG
# -----------------------------
TSO_PATTERNS = {
    "TenneT": "TenneT",
    "50Hertz": "50Hertz",
    "Amprion": "Amprion",
    "TransnetBW": "TransnetBW",
    "TransnetBw": "TransnetBW",
}

OUTPUT_FILE = "tso_net_exports_hourly_wide.csv"

# -----------------------------
def detect_tso(filename):
    name = os.path.basename(filename).lower()
    for k in TSO_PATTERNS:
        if k.lower() in name:
            return TSO_PATTERNS[k]
    raise ValueError(f"Could not detect TSO from filename: {filename}")

def parse_number(s):
    if pd.isna(s) or s == "-" or s == "":
        return np.nan
    return float(str(s).replace(",", ""))

def main():
    files = glob.glob("*.csv")
    if not files:
        raise RuntimeError("No CSV files found")

    all_series = []

    for f in files:
        tso = detect_tso(f)
        print(f"Reading {f} → {tso}")

        df = pd.read_csv(f, sep=";", engine="python")

        # Parse datetime
        df["datetime"] = pd.to_datetime(df["Start date"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # Prefer explicit Net export column if present
        if "Net export [MWh] Calculated resolutions" in df.columns:
            net = df["Net export [MWh] Calculated resolutions"].apply(parse_number)
        else:
            net = pd.Series(np.zeros(len(df)))

        # If Net export column is empty, reconstruct from borders
        if net.isna().mean() > 0.9:
            exp_cols = [c for c in df.columns if "(export)" in c.lower()]
            imp_cols = [c for c in df.columns if "(import)" in c.lower()]

            exp = df[exp_cols].applymap(parse_number).sum(axis=1, skipna=True)
            imp = df[imp_cols].applymap(parse_number).sum(axis=1, skipna=True)

            net = exp - imp

        tmp = pd.DataFrame({
            "datetime": df["datetime"],
            f"netexp_{tso}": net
        })

        all_series.append(tmp)

    # Merge all TSOs on datetime
    out = all_series[0]
    for s in all_series[1:]:
        out = out.merge(s, on="datetime", how="outer")

    out = out.sort_values("datetime").reset_index(drop=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Columns:", list(out.columns))

if __name__ == "__main__":
    main()
