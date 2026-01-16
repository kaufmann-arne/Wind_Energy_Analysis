"""SCADA + Status processing for Greenbyte-style exports.

This module contains the functions you developed for:
- robust header detection in SCADA CSVs that start with commented metadata
- minimal column loading
- Greenbyte status export parsing and building a 10-minute availability mask
- turbine-level hourly aggregation and park-level aggregation

The functions are kept feature-equivalent to your notebook code.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


SCADA_REQUIRED_DEFAULT = {
    "Date and time": "timestamp",
    "Power (kW)": "power_kw",
    "Wind speed (m/s)": "wind_speed_ms",
}


def make_unique(names: list[str]) -> list[str]:
    """Make column names unique by appending .1, .2, ..."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
    return out


def find_scada_header_and_columns(path: Path) -> tuple[int, list[str]]:
    """Locate the commented header line and return (line_index, columns)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("#") and "," in line:
                header = line.lstrip("# ").strip()
                cols = pd.read_csv(StringIO(header), header=None).iloc[0].tolist()
                cols = make_unique(cols)
                return i, cols

    raise ValueError(f"No SCADA header found in {path}")


def load_scada_minimal(
    scada_csv: str | Path,
    required_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load only timestamp, power, and wind speed from Greenbyte SCADA export."""
    required_map = required_map or SCADA_REQUIRED_DEFAULT

    path = Path(scada_csv)
    header_idx, columns = find_scada_header_and_columns(path)

    usecols = [c for c in columns if c in required_map]

    df = pd.read_csv(
        path,
        skiprows=header_idx + 1,
        header=None,
        names=columns,
        usecols=usecols,
        low_memory=True,
    )

    df = df.rename(columns=required_map)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    return df


def read_greenbyte_status_csv(
    path: str | Path,
    datetime_columns: list[str] = ["Timestamp start", "Timestamp end"],
) -> pd.DataFrame:
    """Read Greenbyte Status CSV with robust header detection."""
    path = Path(path)

    header_line_idx: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line_stripped = line.strip()

            if not line_stripped or line_stripped.startswith("#"):
                continue

            if line_stripped.count(",") >= 2:
                header_line_idx = i
                break

    if header_line_idx is None:
        raise ValueError(f"No Status header found in {path}")

    df = pd.read_csv(
        path,
        skiprows=header_line_idx,
        header=0,
        low_memory=False,
    )

    df.columns = df.columns.str.strip()

    for col in datetime_columns:
        if col not in df.columns:
            raise ValueError(f"Missing datetime column '{col}' in {path}")
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def build_status_mask_10min(status_csv: str | Path, turbine_id: str) -> pd.DataFrame:
    """Build a 10-minute availability mask from status intervals.

    Rule (as in your notebook): default available; only intervals with IEC category
    != 'Full Performance' set availability to False.
    """
    df = read_greenbyte_status_csv(status_csv)

    df = df.dropna(subset=["Timestamp start", "Timestamp end"])

    t_start = df["Timestamp start"].min().floor("10min")
    t_end = df["Timestamp end"].max().ceil("10min")

    timeline = pd.date_range(t_start, t_end, freq="10min", tz="UTC")

    mask = pd.DataFrame({"timestamp": timeline, "is_available": True})

    for _, row in df.iterrows():
        if row.get("IEC category") != "Full Performance":
            mask.loc[
                (mask["timestamp"] >= row["Timestamp start"]) &
                (mask["timestamp"] < row["Timestamp end"]),
                "is_available",
            ] = False

    mask["turbine_id"] = turbine_id
    return mask


def aggregate_scada_hourly_greenbyte(scada_csv: str | Path, turbine_id: str) -> pd.DataFrame:
    """Aggregate Greenbyte SCADA to hourly.

    - wind_speed_ms: hourly mean
    - energy_kwh: sum(power_kw)/6 (10-minute power samples)

    This matches your notebook implementation.
    """
    df = load_scada_minimal(scada_csv)

    hourly = (
        df.set_index("timestamp")
        .resample("1h")
        .agg(
            wind_speed_ms=("wind_speed_ms", "mean"),
            energy_kwh=("power_kw", lambda x: x.sum() / 6.0),
        )
        .reset_index()
    )

    hourly["turbine_id"] = turbine_id
    return hourly


def load_turbine_timeseries(
    *,
    turbine_id: str,
    scada_files: list[str | Path],
    status_files: list[str | Path],
) -> pd.DataFrame:
    """Build a single turbine-level hourly dataset across all years."""

    scada_all = [aggregate_scada_hourly_greenbyte(f, turbine_id) for f in scada_files]
    df_scada = pd.concat(scada_all, ignore_index=True)

    status_all = [build_status_mask_10min(f, turbine_id) for f in status_files]
    df_status_10m = pd.concat(status_all, ignore_index=True) if status_all else pd.DataFrame(
        columns=["timestamp", "is_available", "turbine_id"]
    )

    if not df_status_10m.empty:
        status_hourly = (
            df_status_10m.set_index("timestamp")
            .resample("1h")
            .agg(availability_fraction=("is_available", "mean"))
            .reset_index()
        )
        status_hourly["is_available"] = status_hourly["availability_fraction"] >= (2 / 3)
    else:
        status_hourly = pd.DataFrame(columns=["timestamp", "availability_fraction", "is_available"])

    df = df_scada.merge(status_hourly, on="timestamp", how="left")

    # Default missing status to False (unavailable) – matches notebook behavior.
    df["is_available"] = df["is_available"].fillna(False)

    return df.sort_values("timestamp")


def aggregate_park_hourly(df_all_turbines: pd.DataFrame, site_name: str) -> pd.DataFrame:
    """Compute hourly park-level means across all turbines."""
    park_hourly = (
        df_all_turbines
        .groupby("timestamp")
        .agg(
            wind_speed_ms_mean=("wind_speed_ms", "mean"),
            energy_kwh_mean=("energy_kwh", "mean"),
            availability_fraction=("is_available", "mean"),
        )
        .reset_index()
    )

    park_hourly["site_name"] = site_name
    park_hourly["is_available"] = park_hourly["availability_fraction"] >= (2 / 3)

    return park_hourly


def process_site(
    *,
    project_root: str | Path,
    site_name: str,
    turbine_ids: list[str],
    processed_subdir: str = "Data/Processed/Pen_Kel_Pre",
    scada_subdir: str = "Data/Raw/Scada",
) -> None:
    """Process all turbines for one site and store hourly processed data.

    This is a functional equivalent of your notebook `process_site()`.
    """
    project_root = Path(project_root)

    scada_dir = project_root / scada_subdir / f"Scada_{site_name}"

    processed_dir = project_root / processed_subdir / site_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_turbines: list[pd.DataFrame] = []

    for turbine_id in turbine_ids:
        out_path = processed_dir / f"{turbine_id}_hourly.csv"

        if out_path.exists():
            print(f"Skipping {site_name} - {turbine_id} (already processed)")
            df_existing = pd.read_csv(out_path, parse_dates=["timestamp"])
            all_turbines.append(df_existing)
            continue

        print(f"Processing {site_name} - {turbine_id}")

        scada_files = sorted(scada_dir.glob(f"Turbine_Data_{turbine_id}_*.csv"))
        status_files = sorted(scada_dir.glob(f"Status_{turbine_id}_*.csv"))

        if not scada_files:
            print(f"  WARNING: No SCADA files found for {turbine_id}")
            continue

        df_turbine = load_turbine_timeseries(
            turbine_id=turbine_id,
            scada_files=scada_files,
            status_files=status_files,
        )

        df_turbine.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path.relative_to(project_root)}")

        all_turbines.append(df_turbine)

    if all_turbines:
        df_all = pd.concat(all_turbines, ignore_index=True)
        park_hourly = aggregate_park_hourly(df_all, site_name)

        park_out = processed_dir / f"{site_name}_park_hourly.csv"
        park_hourly.to_csv(park_out, index=False)

        print(f"  Park hourly saved -> {park_out.relative_to(project_root)}")
