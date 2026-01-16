"""Build ML-ready datasets by merging SCADA with ERA5.

This module is a direct modularization of your notebook code:
- helper utilities for wind direction, power-law extrapolation, expected energy
- cyclical time encodings
- `build_ml_datasets_per_turbine(...)` that creates one dataset per turbine

The function preserves signature, feature names, and target engineering behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr


def _to_path_list(x: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
    """Ensure a list of file paths (strings)."""
    if isinstance(x, (str, Path)):
        return [str(x)]
    return [str(p) for p in x]


def _wind_direction_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction in degrees FROM which wind blows."""
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def _power_law(ws_ref: np.ndarray, z_ref: float, z_target: float, alpha: np.ndarray) -> np.ndarray:
    """Vertical wind extrapolation using the power law."""
    return ws_ref * (z_target / z_ref) ** alpha


def _expected_energy_from_curve(ws: np.ndarray, power_curve: np.ndarray) -> np.ndarray:
    """Expected hourly energy (kWh) from a turbine power curve.

    ERA5 is hourly, so the interpolated power (kW) numerically equals hourly energy (kWh) for a 1-hour interval.
    """
    return np.interp(
        ws,
        power_curve[:, 0],
        power_curve[:, 1],
        left=0.0,
        right=float(power_curve[:, 1].max()),
    )


def _find_rated_ws(power_curve: np.ndarray, rated_power_kw: float) -> float:
    """First wind speed where power reaches 99% of rated."""
    p = power_curve[:, 1]
    v = power_curve[:, 0]
    idx = np.where(p >= 0.99 * rated_power_kw)[0]
    return float(v[idx[0]]) if len(idx) else float(v.max())


def _cyclical_time_features(ts: pd.Series) -> pd.DataFrame:
    """Cyclical time encodings for ML."""
    out = pd.DataFrame(index=ts.index)
    out["hour"] = ts.dt.hour
    out["doy"] = ts.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def build_ml_datasets_per_turbine(
    *,
    era5_main_paths: Union[str, Path, List[Union[str, Path]]],
    era5_fsr_zust_paths: Union[str, Path, List[Union[str, Path]]],
    scada_csv_path: Union[str, Path],
    turbine_power_curve_json: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    target_tz: str = "UTC",
    merge_tolerance: str = "45min",
    climate_prefix: str = "era5_",
    use_log_target: bool = True,
    min_ws_above_cutin: float = 0.5,
    max_ws_fraction_of_rated: float = 0.90,
    min_expected_energy_kwh: float = 50.0,
) -> Dict[str, pd.DataFrame]:
    """Build one ML-ready dataset per turbine.

    Targets
    -------
    - target_correction_factor
    - target_log_correction_factor (optional)
    - target_is_valid (training mask)

    Climate features are prefixed with `climate_prefix`.
    """

    scada = pd.read_csv(scada_csv_path)
    scada["timestamp"] = pd.to_datetime(scada["timestamp"], utc=True).dt.tz_convert(target_tz)
    scada = scada.sort_values("timestamp").reset_index(drop=True)

    datasets: Dict[str, pd.DataFrame] = {}

    with open(turbine_power_curve_json, "r", encoding="utf-8") as f:
        turbine_curves = json.load(f)

    # xarray combine-by-coords to accept monthly files
    ds_main = xr.open_mfdataset(_to_path_list(era5_main_paths), combine="by_coords")
    ds_fsr_zust = xr.open_mfdataset(_to_path_list(era5_fsr_zust_paths), combine="by_coords")

    # Some exports use valid_time
    if "valid_time" in ds_main:
        ds_main = ds_main.rename({"valid_time": "time"})
    if "valid_time" in ds_fsr_zust:
        ds_fsr_zust = ds_fsr_zust.rename({"valid_time": "time"})

    for turbine_id, df in scada.groupby("turbine_id"):
        df = df.copy().reset_index(drop=True)

        lat = float(df["lat"].iloc[0])
        lon = float(df["lon"].iloc[0])
        hub_height = float(df["hub_height_m"].iloc[0])
        rated_power = float(df["rated_power_kw"].iloc[0])
        turbine_type = str(df["turbine_type"].iloc[0])

        if turbine_type not in turbine_curves:
            raise KeyError(f"Missing power curve for {turbine_type}")

        curve = np.asarray(turbine_curves[turbine_type]["power_curve"], dtype=float)
        cut_in = float(turbine_curves[turbine_type].get("Cut-in wind speed", 3.5))
        rated_ws = _find_rated_ws(curve, rated_power)

        # Spatial interpolation to site
        ds_m = ds_main.interp(latitude=lat, longitude=lon)
        ds_r = ds_fsr_zust.interp(latitude=lat, longitude=lon)

        time_index = pd.to_datetime(ds_m["time"].values, utc=True).tz_convert(target_tz)

        u10, v10 = ds_m["u10"].values, ds_m["v10"].values
        u100, v100 = ds_m["u100"].values, ds_m["v100"].values

        ws10 = np.sqrt(u10**2 + v10**2)
        ws100 = np.sqrt(u100**2 + v100**2)

        alpha = np.log((ws100 + 1e-6) / (ws10 + 1e-6)) / np.log(100 / 10)
        ws_hub = _power_law(ws10, 10.0, hub_height, alpha)

        t2m = ds_m["t2m"].values
        sp = ds_m["sp"].values
        d2m = ds_m["d2m"].values if "d2m" in ds_m else None

        # Air density (humidity correction when d2m exists)
        if d2m is not None:
            e = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
            q = 0.622 * e / (sp - 0.378 * e)
            tv = t2m * (1 + 0.61 * q)
            air_density = sp / (287.05 * tv)
        else:
            air_density = sp / (287.05 * t2m)

        era5_df = pd.DataFrame(
            {
                "timestamp": time_index,
                f"{climate_prefix}ws_10m": ws10,
                f"{climate_prefix}ws_100m": ws100,
                f"{climate_prefix}ws_hub": ws_hub,
                f"{climate_prefix}wd_10m": _wind_direction_from_uv(u10, v10),
                f"{climate_prefix}wd_100m": _wind_direction_from_uv(u100, v100),
                f"{climate_prefix}shear_alpha": alpha,
                f"{climate_prefix}t2m_K": t2m,
                f"{climate_prefix}d2m_K": d2m,
                f"{climate_prefix}sp_Pa": sp,
                f"{climate_prefix}air_density_kgm3": air_density,
                f"{climate_prefix}friction_velocity_ms": ds_r["zust"].values,
                f"{climate_prefix}surface_roughness_m": ds_r["fsr"].values,
            }
        )

        era5_df = pd.concat([era5_df, _cyclical_time_features(era5_df["timestamp"])], axis=1)

        merged = pd.merge_asof(
            df,
            era5_df,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(merge_tolerance),
        )

        merged["expected_energy_kwh"] = _expected_energy_from_curve(
            merged[f"{climate_prefix}ws_hub"].values,
            curve,
        )

        ws_min = cut_in + float(min_ws_above_cutin)
        ws_max = float(max_ws_fraction_of_rated) * rated_ws

        merged["target_is_valid"] = (
            merged["energy_kwh"].notna()
            & merged["expected_energy_kwh"].notna()
            & (merged[f"{climate_prefix}ws_hub"] >= ws_min)
            & (merged[f"{climate_prefix}ws_hub"] <= ws_max)
            & (merged["expected_energy_kwh"] >= float(min_expected_energy_kwh))
        )

        merged["target_correction_factor"] = np.nan
        merged.loc[merged["target_is_valid"], "target_correction_factor"] = (
            merged.loc[merged["target_is_valid"], "energy_kwh"]
            / merged.loc[merged["target_is_valid"], "expected_energy_kwh"]
        )

        if use_log_target:
            merged["target_log_correction_factor"] = np.nan
            valid_log_mask = merged["target_is_valid"] & (merged["target_correction_factor"] > 0)
            merged.loc[valid_log_mask, "target_log_correction_factor"] = np.log(
                merged.loc[valid_log_mask, "target_correction_factor"]
            )

        datasets[str(turbine_id)] = merged

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out / f"ml_dataset_{turbine_id}.csv", index=False)

    return datasets
