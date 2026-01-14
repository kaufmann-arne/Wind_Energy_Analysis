import glob
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
PRICE_GLOB = "eh*.csv"
WIND_GLOB  = "wh*.csv"

OUT_PRICE = "prices_hourly_merged.csv"
OUT_WIND  = "wind_onshore_hourly_merged.csv"

DT_COL = "Start date"  # from your test files

PRICE_COL_OLD = "DE/AT/LU [€/MWh] Calculated resolutions"
PRICE_COL_NEW = "Germany/Luxembourg [€/MWh] Calculated resolutions"
PRICE_SWITCH_DATE = pd.Timestamp("2018-10-01")

WIND_COL = "Wind onshore [MWh] Calculated resolutions"

DT_FORMAT = "%b %d, %Y %I:%M %p"  # e.g. "Dec 17, 2025 12:00 AM"
# ----------------------------------------


def read_smard_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def parse_dt(series: pd.Series) -> pd.Series:
    # Fixed format => no warnings, faster
    return pd.to_datetime(series, format=DT_FORMAT, errors="coerce")


def to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str)

    # Normalize special spaces
    s = s.str.replace("\u00a0", " ", regex=False).str.replace("\u202f", " ", regex=False)
    s = s.str.strip()

    # Treat common empties as NaN
    s = s.replace({"": np.nan, "-": np.nan, "–": np.nan, "nan": np.nan, "None": np.nan})

    # Keep only digits, minus, comma, dot
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)

    # If both comma and dot exist: decide which is decimal by last occurrence
    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains(r"\.", na=False)
    both = has_comma & has_dot

    # EU style: 1.234,56  -> remove dots (thousands), comma -> dot
    eu = both & (s.str.rfind(",") > s.str.rfind("."))
    s = s.where(~eu, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))

    # US style: 1,234.56 -> remove commas (thousands)
    us = both & ~eu
    s = s.where(~us, s.str.replace(",", "", regex=False))

    # Only comma present -> assume comma decimal
    only_comma = has_comma & ~has_dot
    s = s.where(~only_comma, s.str.replace(",", ".", regex=False))

    return pd.to_numeric(s, errors="coerce")


def merge_and_dedupe(parts: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["datetime", value_col])
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    return df


def build_prices() -> pd.DataFrame:
    files = sorted(glob.glob(PRICE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {PRICE_GLOB}")

    parts = []
    for fp in files:
        raw = read_smard_csv(fp)
        raw["datetime"] = parse_dt(raw[DT_COL])
        raw = raw[raw["datetime"].notna()].copy()

        price = pd.Series(np.nan, index=raw.index, dtype="float64")

        if PRICE_COL_OLD in raw.columns:
            m_old = raw["datetime"] < PRICE_SWITCH_DATE
            price.loc[m_old] = to_float(raw.loc[m_old, PRICE_COL_OLD])

        if PRICE_COL_NEW in raw.columns:
            m_new = raw["datetime"] >= PRICE_SWITCH_DATE
            price.loc[m_new] = to_float(raw.loc[m_new, PRICE_COL_NEW])

        out = pd.DataFrame({"datetime": raw["datetime"], "price_eur_mwh": price})
        out = out[out["price_eur_mwh"].notna()]
        print(f"[PRICE] {fp}: kept {len(out):,} rows")
        parts.append(out)

    return merge_and_dedupe(parts, "price_eur_mwh")


def build_wind() -> pd.DataFrame:
    files = sorted(glob.glob(WIND_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {WIND_GLOB}")

    parts = []
    for fp in files:
        raw = read_smard_csv(fp)
        raw["datetime"] = parse_dt(raw[DT_COL])
        raw = raw[raw["datetime"].notna()].copy()

        out = pd.DataFrame({
            "datetime": raw["datetime"],
            "wind_onshore_mwh": to_float(raw[WIND_COL]),
        })

        out = out[out["wind_onshore_mwh"].notna()]
        print(f"[WIND ] {fp}: kept {len(out):,} rows")
        parts.append(out)

    return merge_and_dedupe(parts, "wind_onshore_mwh")


def main():
    prices = build_prices()
    prices.to_csv(OUT_PRICE, index=False)
    print(f"\nWrote {len(prices):,} rows -> {OUT_PRICE}")

    wind = build_wind()
    wind.to_csv(OUT_WIND, index=False)
    print(f"Wrote {len(wind):,} rows -> {OUT_WIND}")


if __name__ == "__main__":
    main()
