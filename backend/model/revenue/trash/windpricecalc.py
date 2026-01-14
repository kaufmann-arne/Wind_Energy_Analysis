import os
import pandas as pd

# -----------------------------
# Config (edit filenames)
# -----------------------------
PRICE_FILE = "prices_hourly_merged.csv"          # <- your hourly price file
WIND_FILE  = "wind_onshore_hourly_merged.csv"   # <- your hourly wind file
OUTPUT_FILE = "smard_calc_wind_onshore_monthly.csv"

# If your script + csvs are in the same folder, this makes paths robust
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_PATH = os.path.join(BASE_DIR, PRICE_FILE)
WIND_PATH  = os.path.join(BASE_DIR, WIND_FILE)
OUT_PATH   = os.path.join(BASE_DIR, OUTPUT_FILE)

def main():
    # Read
    prices = pd.read_csv(PRICE_PATH, parse_dates=["datetime"])
    wind   = pd.read_csv(WIND_PATH,  parse_dates=["datetime"])

    # Basic checks
    if "price_eur_mwh" not in prices.columns:
        raise ValueError("price file must have column: price_eur_mwh")
    if "wind_onshore_mwh" not in wind.columns:
        raise ValueError("wind file must have column: wind_onshore_mwh")

    # Drop duplicates if any (keep last)
    prices = prices.sort_values("datetime").drop_duplicates("datetime", keep="last")
    wind   = wind.sort_values("datetime").drop_duplicates("datetime", keep="last")

    # Merge on hour (inner join ensures both available)
    df = prices.merge(wind, on="datetime", how="inner")

    # Remove rows with missing values
    df = df.dropna(subset=["price_eur_mwh", "wind_onshore_mwh"])

    # Optional: if there are any negative/zero wind values, handle them
    df = df[df["wind_onshore_mwh"] > 0]

    # Create month key (first day of month)
    df["date"] = df["datetime"].dt.to_period("M").dt.to_timestamp()

    # Wind-weighted monthly price
    # P_wind = sum(price * wind) / sum(wind)
    grouped = df.groupby("date", as_index=False).apply(
        lambda g: pd.Series({
            "wind_onshore_eur_mwh_calc": (g["price_eur_mwh"] * g["wind_onshore_mwh"]).sum() / g["wind_onshore_mwh"].sum(),
            "wind_onshore_mwh_sum": g["wind_onshore_mwh"].sum(),
            "hours_used": len(g)
        })
    )

    # Clean index created by groupby+apply
    grouped = grouped.reset_index(drop=True)

    # Save
    grouped.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with {len(grouped)} months.")

    # Quick sanity print
    print(grouped.head(3).to_string(index=False))
    print(grouped.tail(3).to_string(index=False))

if __name__ == "__main__":
    main()
