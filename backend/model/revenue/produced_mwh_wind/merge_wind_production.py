import glob
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
WIND_GLOB = "wh*.csv"
OUT_WIND = "wind_onshore_hourly_merged.csv"

DT_COL = "Start date"
WIND_COL = "Wind onshore [MWh] Calculated resolutions"

DT_FORMAT = "%b %d, %Y %I:%M %p"
# ----------------------------------------


def read_smard_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DT_FORMAT, errors="coerce")


def to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str)

    s = s.str.replace("\u00a0", " ", regex=False).str.replace("\u202f", " ", regex=False)
    s = s.str.strip()

    s = s.replace({"": np.nan, "-": np.nan, "–": np.nan, "nan": np.nan, "None": np.nan})

    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)

    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains(r"\.", na=False)
    both = has_comma & has_dot

    eu = both & (s.str.rfind(",") > s.str.rfind("."))
    s = s.where(~eu, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))

    us = both & ~eu
    s = s.where(~us, s.str.replace(",", "", regex=False))

    only_comma = has_comma & ~has_dot
    s = s.where(~only_comma, s.str.replace(",", ".", regex=False))

    return pd.to_numeric(s, errors="coerce")


def merge_and_dedupe(parts: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["datetime", value_col])
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    return df


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
    wind = build_wind()
    wind.to_csv(OUT_WIND, index=False)
    print(f"\nWrote {len(wind):,} rows -> {OUT_WIND}")


if __name__ == "__main__":
    main()
