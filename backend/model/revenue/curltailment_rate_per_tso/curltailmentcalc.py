import os
import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRODUCTION_FILE = "wind_onshore_daily_by_tso.csv"
CURTAILMENT_FILE = "curtailment.csv"

OUTPUT_FILE = "curtailment_rate_quarterly_by_tso.csv"

# TSO column names (must match both files)
TSOS = ["TenneT", "50Hertz", "Amprion", "TransnetBW"]

# -------------------------------------------------
def parse_german_number(x):
    """
    Convert strings like '863,24' to float 863.24
    """
    if pd.isna(x):
        return float("nan")
    return float(str(x).replace(".", "").replace(",", "."))

def parse_quarter_label(q):
    """
    'Q1 2015' -> pandas Period('2015Q1')
    """
    q = q.strip()
    year = int(q[-4:])
    quarter = int(q[1])
    return pd.Period(year=year, quarter=quarter, freq="Q")

def main():
    # -----------------------------
    # Load production data (daily)
    # -----------------------------
    prod = pd.read_csv(
        os.path.join(BASE_DIR, PRODUCTION_FILE),
        parse_dates=["date"]
    )

    # Ensure numeric
    for tso in TSOS:
        prod[tso] = pd.to_numeric(prod[tso], errors="coerce")

    # Create quarter index
    prod["quarter"] = prod["date"].dt.to_period("Q")

    # Aggregate daily → quarterly (MWh)
    prod_q = (
        prod.groupby("quarter")[TSOS]
        .sum()
        .reset_index()
    )

    # -----------------------------
    # Load curtailment data (quarterly)
    # -----------------------------
    curt = pd.read_csv(os.path.join(BASE_DIR, CURTAILMENT_FILE))

    # Parse quarter
    curt["quarter"] = curt["Quarter"].apply(parse_quarter_label)

    # Parse German numbers & convert GWh → MWh
    for tso in TSOS:
        curt[tso] = curt[tso].apply(parse_german_number) * 1_000

    # Keep only relevant columns
    curt_q = curt[["quarter"] + TSOS]

    # -----------------------------
    # Merge production + curtailment
    # -----------------------------
    df = prod_q.merge(curt_q, on="quarter", how="inner", suffixes=("_prod", "_curt"))

    # -----------------------------
    # Compute curtailment rates
    # -----------------------------
    out = pd.DataFrame({"quarter": df["quarter"]})

    for tso in TSOS:
        prod_col = f"{tso}_prod"
        curt_col = f"{tso}_curt"

        out[f"{tso}_curtailment_rate"] = (
            df[curt_col] / (df[curt_col] + df[prod_col])
        )

        out[f"{tso}_production_mwh"] = df[prod_col]
        out[f"{tso}_curtailed_mwh"] = df[curt_col]

    # Sort and save
    out = out.sort_values("quarter").reset_index(drop=True)
    out.to_csv(os.path.join(BASE_DIR, OUTPUT_FILE), index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print("\nPreview:")
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
