from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
import math
import pandas as pd


def split_loegtved_txt_to_turbine_csvs(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    timestamp_col: str = "PCTimeStamp",
    turbine_prefix_regex: str = r"(WTG\d+)_",
    sep: str = ";",
    decimal: str = ",",
) -> list[Path]:
    """
    Matches your notebook:
    - reads one wide TXT/CSV with columns like WTG01_..., WTG02_...
    - cleans PCTimeStamp:
        * date-only -> add time
        * 00.10.00 -> 00:10:00
    - writes one CSV per turbine: WTG01_SCADA.csv, ...
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        input_path,
        sep=sep,
        decimal=decimal,
        skipinitialspace=True,
        engine="python",
    )

    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^$")]

    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column '{timestamp_col}' in {input_path.name}")

    ts = df[timestamp_col].astype(str).str.strip()
    ts = ts.str.replace(r"\s+", " ", regex=True) 
    # Wenn nur Datum vorhanden -> Zeit ergänzen
    # erkennt sowohl "hh.mm.ss" als auch "hh:mm:ss" und auch einstelliges hh
    has_time = ts.str.contains(r"\b\d{1,2}[.:]\d{2}[.:]\d{2}\b", regex=True)
    ts = ts.where(has_time, ts + " 00.00.00")

    # Zeitseparatoren vereinheitlichen: 00.10.00 -> 00:10:00 (auch 0.10.00)
    ts = ts.str.replace(r"\b(\d{1,2})\.(\d{2})\.(\d{2})\b", r"\1:\2:\3", regex=True)

    # Optional (stabiler): explizites Format nach Normalisierung versuchen
    df[timestamp_col] = pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df[timestamp_col] = (
        df[timestamp_col]
        .dt.tz_localize("Europe/Copenhagen", ambiguous=True, nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    if df[timestamp_col].isna().any():
        bad = df.loc[df[timestamp_col].isna(), timestamp_col].head(5).tolist()
        raise ValueError(f"Timestamp parsing produced NaT. Examples: {bad}")

    turbines = sorted(
        {
            re.match(turbine_prefix_regex, c).group(1)
            for c in df.columns
            if re.match(turbine_prefix_regex, c)
        }
    )
    if not turbines:
        raise ValueError(f"No turbine prefixes found with regex '{turbine_prefix_regex}'")

    out_files: list[Path] = []
    for wtg in turbines:
        wtg_cols = [c for c in df.columns if c.startswith(f"{wtg}_")]
        df_wtg = df[[timestamp_col] + wtg_cols].copy()

        df_wtg.columns = [timestamp_col] + [c.replace(f"{wtg}_", "").strip() for c in wtg_cols]

        out_file = output_dir / f"{wtg}_SCADA.csv"
        df_wtg.to_csv(out_file, index=False, sep=",", date_format="%Y-%m-%d %H:%M:%S%z")
        out_files.append(out_file)

    return out_files


def aggregate_loegtved_turbine_hourly(
    data: str | Path | pd.DataFrame,
    *,
    timestamp_col: str = "PCTimeStamp",
    alarm_col: str = "System Logs First Active Alarm No",
    power_col_kw: str = "Grid Production Power Avg.",
    windspeed_col: str = "Ambient WindSpeed Estimated Avg.",
    min_valid_10min_per_hour: int = 4,
    availability_fraction: float = 2.0 / 3.0,
) -> pd.DataFrame:
    """
    Hourly aggregation with 'valid measurements' logic:
    - valid_10min: alarm, power, wind all present and power is numeric
    - is_available_10min: valid_10min and alarm == 0
    - energy_kwh_10min: power_kw / 6 (only meaningful if power numeric)
    - Keep only hours with >= min_valid_10min_per_hour valid frames
    - Hour is_available: available_10min >= availability_fraction * n_valid_10min
    """

    if isinstance(data, (str, Path)):
        df = pd.read_csv(data, parse_dates=[timestamp_col])
    else:
        df = data.copy()

    # remove "(1)" suffixes and trim
    df.columns = df.columns.str.replace(r"\s*\(\d+\)", "", regex=True).str.strip()

    needed = [timestamp_col, alarm_col, power_col_kw, windspeed_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in Loegtved turbine SCADA: {missing}")

    # ensure proper dtypes
    df[power_col_kw] = pd.to_numeric(df[power_col_kw], errors="coerce")
    df[alarm_col] = pd.to_numeric(df[alarm_col], errors="coerce")
    df[windspeed_col] = pd.to_numeric(df[windspeed_col], errors="coerce")

    # valid 10-min frame = all required signals present
    df["valid_10min"] = df[[alarm_col, power_col_kw, windspeed_col]].notna().all(axis=1).astype(int)

    # availability at 10-min level: valid AND no alarm
    df["is_available_10min"] = ((df["valid_10min"] == 1) & (df[alarm_col] == 0)).astype(int)

    # energy per 10-min (kWh)
    df["energy_kwh_10min"] = df[power_col_kw] / 6.0

    df = df.set_index(timestamp_col)

    hourly = df.resample("1h").agg(
        energy_kwh=("energy_kwh_10min", lambda s: s.sum(min_count=1)),
        windspeed_mean=(windspeed_col, "mean"),
        available_10min=("is_available_10min", "sum"),
        n_valid_10min=("valid_10min", "sum"),
    )

    # keep only hours with enough valid frames
    hourly = hourly[hourly["n_valid_10min"] >= int(min_valid_10min_per_hour)]

    # hourly availability decision (based only on valid frames)
    hourly["is_available"] = (
        hourly["available_10min"] >= availability_fraction * hourly["n_valid_10min"]
    ).astype(int)

    return hourly.reset_index()




def aggregate_loegtved_park_hourly(
    hourly_all: pd.DataFrame,
    *,
    timestamp_col: str = "PCTimeStamp",
    n_turbines_total: int,
    availability_fraction_threshold: float = 2.0 / 3.0,
) -> pd.DataFrame:
    """
    Park aggregation (fixed denominator = total turbines in park):

    - available_turbines = sum(is_available) across turbines present that hour
      (turbines missing that hour implicitly count as unavailable)
    - availability_fraction = available_turbines / n_turbines_total
    - is_available = availability_fraction >= threshold  (equiv. available_turbines >= ceil(threshold*n_turbines_total))

    - park energy/wind are computed ONLY across available turbines (as mean, per your request)
    """

    if n_turbines_total <= 0:
        raise ValueError("n_turbines_total must be a positive integer")

    needed = [timestamp_col, "energy_kwh", "windspeed_mean", "is_available", "turbine"]
    missing = [c for c in needed if c not in hourly_all.columns]
    if missing:
        raise KeyError(f"Missing required columns for park aggregation: {missing}")

    df = hourly_all.copy()
    df["is_available"] = df["is_available"].astype("boolean").fillna(False)

    # Count available turbines per hour (missing turbines are not in df -> treated as unavailable via fixed denominator)
    avail_count = (
        df.groupby(timestamp_col)["is_available"]
        .sum()
        .rename("available_turbines")
    )

    # Diagnostics: how many turbines actually have data that hour (not used as denominator)
    with_data = (
        df.groupby(timestamp_col)["turbine"]
        .nunique()
        .rename("turbines_with_data")
    )

    # Mean only across available turbines
    av = df[df["is_available"]].copy()
    park_vals = (
        av.groupby(timestamp_col)
        .agg(
            energy_kwh_park=("energy_kwh", "mean"),
            windspeed_mean=("windspeed_mean", "mean"),
        )
    )

    park = (
        park_vals.join([avail_count, with_data], how="outer")
        .reset_index()
    )

    park["n_turbines_total"] = int(n_turbines_total)

    # Fill counts for hours where no turbines are present
    park["available_turbines"] = park["available_turbines"].fillna(0).astype(int)
    park["turbines_with_data"] = park["turbines_with_data"].fillna(0).astype(int)

    # Fixed denominator fraction + park availability flag
    park["availability_fraction"] = park["available_turbines"] / float(n_turbines_total)

    required = math.ceil(availability_fraction_threshold * n_turbines_total)
    park["is_available"] = park["available_turbines"] >= required

    return park
