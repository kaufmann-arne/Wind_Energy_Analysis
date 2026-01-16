"""Generic SCADA preprocessing into a unified hourly dataset.

This module contains the feature-complete version of your final `preprocess_scada_dataset`
implementation, including:

- Robust timestamp parsing
- Automatic detection of temporal resolution (10-min vs hourly)
- Optional QC column filtering
- Availability derivation when no QC is provided (10-min only)
- Hourly aggregation of 10-min SCADA to energy_kwh with a 2/3 availability rule
- Turbine metadata enrichment and specific power computation

The function is written to be reusable across sites by passing column names and metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def compute_specific_power(rated_power_kw: float, rotor_diameter_m: float) -> float:
    """Compute specific power in W/m^2."""
    if rated_power_kw is None or rotor_diameter_m is None:
        raise ValueError("rated_power_kw and rotor_diameter_m must not be None")
    area = np.pi * (float(rotor_diameter_m) / 2.0) ** 2
    return (float(rated_power_kw) * 1000.0) / area


def preprocess_scada_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    site_id: str,
    site_lat: float,
    site_lon: float,
    timestamp_col: str,
    target_col: str,
    *,
    turbine_col: Optional[str] = None,
    turbine_meta: Optional[dict[str, dict[str, Any]]] = None,
    turbine_db_path: Optional[str | Path] = None,
    windspeed_col: Optional[str] = None,
    qc_col: Optional[str] = None,
    min_power: float = 0.0,
    max_power: Optional[float] = None,
    default_hub_height: Optional[float] = None,
) -> pd.DataFrame:
    """Preprocess raw SCADA into a standardized hourly dataset.

    Parameters
    ----------
    input_dir:
        Path to a directory with CSV files or to a single CSV file.
    output_dir:
        Directory where `scada_<site_id>_processed.csv` is written.
    timestamp_col, target_col, windspeed_col:
        Column names as present in the raw SCADA file(s).
    qc_col:
        If provided, rows are filtered to qc_col == 1. In that case, availability
        derivation from SCADA is skipped.

    Returns
    -------
    pd.DataFrame
        Standardized dataset with columns:
        timestamp, turbine_id, site_id, energy_kwh, wind_speed_ms, lat, lon,
        hub_height_m, rated_power_kw, rotor_diameter_m, specific_power_wpm2, turbine_type
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    turbine_meta = turbine_meta or {}

    allowed_columns = [
        "timestamp",
        "turbine_id",
        "site_id",
        "energy_kwh",
        "wind_speed_ms",
        "lat",
        "lon",
        "hub_height_m",
        "rated_power_kw",
        "rotor_diameter_m",
        "specific_power_wpm2",
        "turbine_type",
    ]

    if turbine_db_path is None:
        raise ValueError("turbine_db_path must be provided")

    with Path(turbine_db_path).open("r", encoding="utf-8") as f:
        turbine_db = json.load(f)

    if input_dir.is_file():
        files = [input_dir]
    else:
        files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise ValueError(f"No CSV files found in {input_dir}")

    all_dfs: list[pd.DataFrame] = []

    for i, file in enumerate(files, start=1):
        df = pd.read_csv(file)

        # Normalize column names (critical for odd whitespace / nbsp)
        df.columns = df.columns.str.strip().str.replace("\u00A0", " ", regex=False)

        # Timestamp
        if timestamp_col not in df.columns:
            raise KeyError(f"Missing timestamp column '{timestamp_col}' in {file.name}")

        df["timestamp"] = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
            dayfirst=True,
            format="mixed",
            utc=True,
        )
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Detect temporal resolution (median timedelta)
        deltas = df["timestamp"].diff().dropna()
        median_minutes = deltas.dt.total_seconds().median() / 60.0

        is_10min = 5 <= median_minutes <= 15
        is_hourly = 45 <= median_minutes <= 75

        if not (is_10min or is_hourly):
            # keep conservative behavior: allow, but warn via ValueError message
            # (caller can decide to adjust thresholds)
            raise ValueError(
                f"Could not classify temporal resolution for {file.name}. "
                f"Median step: {median_minutes:.2f} minutes"
            )

        # Turbine id
        if turbine_col is None:
            turbine_id = f"{site_id}_T{i:02d}"
        else:
            if turbine_col not in df.columns:
                raise KeyError(f"Missing turbine_col '{turbine_col}' in {file.name}")
            turbine_id = str(df[turbine_col].iloc[0])

        df["turbine_id"] = turbine_id
        df["site_id"] = site_id

        # Metadata/spec
        meta = turbine_meta.get(turbine_id)
        if meta is None:
            raise ValueError(f"No turbine_meta entry for turbine_id='{turbine_id}'")

        turbine_type = meta.get("turbine_type")
        if turbine_type not in turbine_db:
            raise KeyError(f"Turbine type '{turbine_type}' not found in turbine_db")

        spec = turbine_db[turbine_type]

        cut_in_ws = spec.get("Cut-in wind speed")
        if cut_in_ws is None:
            raise ValueError(f"Cut-in wind speed missing for turbine type '{turbine_type}'")

        df["turbine_type"] = turbine_type
        df["lat"] = meta.get("lat", site_lat)
        df["lon"] = meta.get("lon", site_lon)
        df["hub_height_m"] = meta.get(
            "hub_height_m",
            spec.get("default_hub_height_m", default_hub_height),
        )

        df["rated_power_kw"] = float(spec["rated_power_kw"])
        df["rotor_diameter_m"] = float(spec["rotor_diameter_m"])
        df["specific_power_wpm2"] = compute_specific_power(
            spec["rated_power_kw"], spec["rotor_diameter_m"]
        )

        # Target
        if target_col not in df.columns:
            raise KeyError(f"Missing target column '{target_col}' in {file.name}")

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        if is_hourly:
            # Target is already energy (kWh)
            df["energy_kwh"] = df[target_col]
        else:
            # Sub-hourly: target is power (kW)
            df["power_kw"] = df[target_col]

        # Wind speed (always required in your final design)
        if windspeed_col is None:
            raise ValueError("windspeed_col must be provided to derive availability")
        if windspeed_col not in df.columns:
            raise KeyError(
                f"Windspeed column '{windspeed_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        df["wind_speed_ms"] = pd.to_numeric(df[windspeed_col], errors="coerce")

        # QC / availability logic
        if qc_col is not None:
            if qc_col not in df.columns:
                raise KeyError(f"QC column '{qc_col}' not found in SCADA data")

            before = len(df)
            df = df[df[qc_col] == 1].copy()
            after = len(df)
            df["is_available"] = True

            print(
                f"[QC FILTER] Removed {before - after} rows "
                f"({100 * (before - after) / max(before, 1):.2f}%) "
                f"based on qc_col='{qc_col}'"
            )

        elif is_10min:
            # Availability derivation only for 10-minute data and only if no qc_col
            ws = df["wind_speed_ms"]
            df["is_available"] = ~(
                (df["power_kw"] < 0)
                | (df["power_kw"].isna())
                | ((ws >= cut_in_ws) & (df["power_kw"] <= 0))
            )

        # Basic sanity filtering
        if is_hourly:
            df = df.dropna(subset=["energy_kwh"])
            df = df[df["energy_kwh"] >= 0]
            if max_power is not None:
                # Maximum possible hourly energy is approx rated_power_kw * 1h
                max_energy_kwh = float(max_power)
                df = df[df["energy_kwh"] <= max_energy_kwh]
        else:
            df = df.dropna(subset=["power_kw"])
            df = df[df["power_kw"] >= float(min_power)]
            if max_power is not None:
                df = df[df["power_kw"] <= float(max_power)]

        # Aggregate 10-min to hourly with 2/3 availability rule
        if is_10min:
            df = df.set_index("timestamp")

            def hourly_aggregate(g: pd.DataFrame):
                if g.empty:
                    return None
                if "is_available" not in g.columns:
                    return None

                availability_fraction = g["is_available"].mean()
                if availability_fraction < (2.0 / 3.0):
                    return None

                # 10-min power -> hourly energy (kWh)
                energy_kwh = g["power_kw"].sum() / 6.0

                return pd.Series(
                    {
                        "energy_kwh": energy_kwh,
                        "wind_speed_ms": g["wind_speed_ms"].mean(),
                        "is_available": True,
                        "lat": g["lat"].iloc[0],
                        "lon": g["lon"].iloc[0],
                        "hub_height_m": g["hub_height_m"].iloc[0],
                        "rated_power_kw": g["rated_power_kw"].iloc[0],
                        "rotor_diameter_m": g["rotor_diameter_m"].iloc[0],
                        "specific_power_wpm2": g["specific_power_wpm2"].iloc[0],
                        "turbine_type": g["turbine_type"].iloc[0],
                    }
                )

            df = (
                df.groupby(["site_id", "turbine_id"])
                .resample("1h")
                .apply(hourly_aggregate)
                .dropna()
                .reset_index()
            )

        else:
            df = df.reset_index(drop=True)

        # Final columns
        missing = set(allowed_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns after processing: {missing}")

        df = df[allowed_columns].copy()
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)

    out_path = output_dir / f"scada_{site_id}_processed.csv"
    final_df.to_csv(out_path, index=False)

    print(f"SCADA processed: {site_id}")
    print(f"Files: {len(files)}")
    print(f"Rows: {len(final_df):,}")
    print(f"Output: {out_path}")

    return final_df
