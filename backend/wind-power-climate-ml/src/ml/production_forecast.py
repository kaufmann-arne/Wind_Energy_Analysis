"""Production-time (website) inference and long-horizon monthly forecasting.

This module implements the *final* pipeline you shared for the website:

1) Build a historical lookup table of **monthly** ML-corrected energy production
   using ERA5 climate data and the trained correction-factor model.

2) Forecast the next N years at monthly resolution via a lightweight
   Monte Carlo re-sampling approach (sampling historical monthly energies
   conditioned on calendar month).

Design notes
------------
* Feature engineering matches the training logic in :mod:`src.ml.dataset_builder`:
  - wind speed at 10m/100m, shear alpha, extrapolated hub wind speed
  - wind direction from u/v
  - cyclical time features (hour/doy sine/cosine)
  - air density (humidity-corrected if d2m is available)
  - surface roughness (fsr) and friction velocity (zust)

* The model predicts **log correction factor**; energy is reconstructed as:
    expected_energy_kwh * exp(pred_log_cf)
  where expected_energy_kwh is derived from the turbine power curve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# Helpers (must match training logic)
# ============================================================


def wind_direction_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction (degrees FROM which the wind blows)."""

    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def compute_alpha(ws10: np.ndarray, ws100: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Power-law shear exponent derived from wind speeds at 10m and 100m."""

    return np.log((ws100 + eps) / (ws10 + eps)) / np.log(100.0 / 10.0)


def extrapolate_ws_to_hub(ws10: np.ndarray, hub_height_m: float, alpha: np.ndarray) -> np.ndarray:
    """Extrapolate 10m wind speed to hub height using the power law."""

    return ws10 * (hub_height_m / 10.0) ** alpha


def cyclical_time_features(ts_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical time encodings for hour-of-day and day-of-year."""

    out = pd.DataFrame(index=ts_utc)
    out["hour"] = ts_utc.hour
    out["doy"] = ts_utc.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out[["hour_sin", "hour_cos", "doy_sin", "doy_cos"]]


def expected_power_kw_from_curve(ws: np.ndarray, curve_ws: np.ndarray, curve_p_kw: np.ndarray) -> np.ndarray:
    """Interpolate a turbine power curve (kW) at wind speed ws (m/s)."""

    return np.interp(ws, curve_ws, curve_p_kw, left=0.0, right=float(np.max(curve_p_kw)))


def air_density_from_sp_t2m_d2m(
    sp_pa: np.ndarray,
    t2m_k: np.ndarray,
    d2m_k: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Air density (kg/m^3).

    Matches the training logic in src/ml/dataset_builder.py:
    - if d2m is available, approximate humidity influence via virtual temperature
    - otherwise fall back to dry-air ideal gas
    """

    if d2m_k is None:
        return sp_pa / (287.05 * t2m_k)

    # Vapour pressure (Pa) from dew point (Magnus-type approximation)
    e = 611.2 * np.exp(17.67 * (d2m_k - 273.15) / (d2m_k - 29.65))
    q = 0.622 * e / (sp_pa - 0.378 * e)
    tv = t2m_k * (1.0 + 0.61 * q)
    return sp_pa / (287.05 * tv)


def _open_era5_dir(nc_dir: Path, *, engine: Optional[str] = None) -> xr.Dataset:
    files = sorted(Path(nc_dir).glob("*.nc"))
    if not files:
        raise ValueError(f"No .nc files found in: {nc_dir}")
    kwargs = dict(combine="by_coords")
    if engine is not None:
        kwargs["engine"] = engine
    ds = xr.open_mfdataset(files, **kwargs)
    if "valid_time" in ds:
        ds = ds.rename({"valid_time": "time"})
    return ds


@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    imputer: object
    feature_cols: list[str]


def load_model_artifacts(model_dir: Path) -> ModelArtifacts:
    """Load model, imputer, and feature column order."""

    model = joblib.load(Path(model_dir) / "final_model.pkl")
    imputer = joblib.load(Path(model_dir) / "feature_imputer.pkl")
    with open(Path(model_dir) / "feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return ModelArtifacts(model=model, imputer=imputer, feature_cols=feature_cols)


# ============================================================
# 1) Historical monthly energies (ML-corrected)
# ============================================================


def build_hist_monthly_energy_lookup(
    *,
    lat: float,
    lon: float,
    hub_height_m: float,
    turbine_type: str,
    era5_main_dir: Path,
    turbine_power_curve_json: Path,
    model_dir: Path,
    # Optional: provide d2m as a separate ERA5 directory (as in your website script)
    era5_d2m_dir: Optional[Path] = None,
    # Optional: override engine for xarray (e.g. "netcdf4")
    xarray_engine: Optional[str] = None,
    # Require at least this many hourly values to accept a monthly sum
    min_hours_per_month: int = 24 * 28,
) -> pd.DataFrame:
    """Compute ML-corrected historical monthly energy at a location.

    Returns a DataFrame with columns:
        year, month, energy_kwh
    One row per historical (year, month).
    """

    artifacts = load_model_artifacts(model_dir)

    # ---- power curve + turbine meta
    with open(turbine_power_curve_json, "r", encoding="utf-8") as f:
        curves = json.load(f)
    if turbine_type not in curves:
        raise KeyError(f"Turbine type not found in turbine_power_curve_json: {turbine_type}")

    meta = curves[turbine_type]
    curve = np.asarray(meta["power_curve"], dtype=float)
    curve_ws = curve[:, 0]
    curve_p_kw = curve[:, 1]

    rated_power_kw = float(meta["rated_power_kw"])
    rotor_diameter_m = float(meta["rotor_diameter_m"])
    area = np.pi * (rotor_diameter_m / 2.0) ** 2
    specific_power_wpm2 = (rated_power_kw * 1000.0) / area

    # ---- load ERA5
    ds_main = _open_era5_dir(Path(era5_main_dir), engine=xarray_engine)

    if era5_d2m_dir is not None:
        ds_d2m = _open_era5_dir(Path(era5_d2m_dir), engine=xarray_engine)
        if "d2m" not in ds_d2m:
            raise KeyError(f"Variable 'd2m' not found in dataset from: {era5_d2m_dir}")
        ds_d2m = ds_d2m[["d2m"]]

        # Merge explicitly to ensure we have d2m available
        ds = xr.merge([ds_main, ds_d2m], join="outer", compat="override")
    else:
        ds = ds_main

    # ---- spatial interpolation
    ds_i = ds.interp(latitude=float(lat), longitude=float(lon))

    # ---- time index
    ts = pd.to_datetime(ds_i["time"].values, utc=True)
    ts_index = pd.DatetimeIndex(ts, name="timestamp")
    tf = cyclical_time_features(ts_index)

    # ---- arrays (required vars)
    required = ["u10", "v10", "u100", "v100", "t2m", "sp"]
    missing = [v for v in required if v not in ds_i]
    if missing:
        raise KeyError(f"Missing required ERA5 variables in main dataset: {missing}")

    u10, v10 = ds_i["u10"].values, ds_i["v10"].values
    u100, v100 = ds_i["u100"].values, ds_i["v100"].values
    t2m = ds_i["t2m"].values
    sp = ds_i["sp"].values

    d2m = ds_i["d2m"].values if "d2m" in ds_i else None

    # Optional surface/turbulence vars (but required by your feature list)
    # If not present, raise with a clear message.
    if "fsr" not in ds_i or "zust" not in ds_i:
        raise KeyError(
            "ERA5 variables 'fsr' and/or 'zust' are missing. "
            "Your feature set requires surface_roughness_m (fsr) and friction_velocity_ms (zust). "
            "Ensure your ERA5 main dataset contains these variables (or extend this function to load them separately)."
        )
    fsr = ds_i["fsr"].values
    zust = ds_i["zust"].values

    ws10 = np.sqrt(u10**2 + v10**2)
    ws100 = np.sqrt(u100**2 + v100**2)
    alpha = compute_alpha(ws10, ws100)
    ws_hub = extrapolate_ws_to_hub(ws10, float(hub_height_m), alpha)

    wd10 = wind_direction_from_uv(u10, v10)
    wd100 = wind_direction_from_uv(u100, v100)

    air_density = air_density_from_sp_t2m_d2m(sp, t2m, d2m)

    expected_kw = expected_power_kw_from_curve(ws_hub, curve_ws, curve_p_kw)
    expected_kwh = expected_kw  # ERA5 is hourly

    # ---- feature frame (all hours, vectorized)
    feat = pd.DataFrame(
        {
            # Wind
            "era5_ws_10m": ws10,
            "era5_ws_100m": ws100,
            "era5_ws_hub": ws_hub,
            "era5_wd_10m": wd10,
            "era5_wd_100m": wd100,
            "era5_shear_alpha": alpha,

            # Thermodynamics
            "era5_t2m_K": t2m,
            "era5_d2m_K": d2m,
            "era5_sp_Pa": sp,
            "era5_air_density_kgm3": air_density,

            # Surface/turbulence
            "era5_surface_roughness_m": fsr,
            "era5_friction_velocity_ms": zust,
        },
        index=ts_index,
    )

    feat = pd.concat([feat, tf], axis=1)

    # static features
    feat["hub_height_m"] = float(hub_height_m)
    feat["rated_power_kw"] = float(rated_power_kw)
    feat["rotor_diameter_m"] = float(rotor_diameter_m)
    feat["specific_power_wpm2"] = float(specific_power_wpm2)
    feat["lat"] = float(lat)
    feat["lon"] = float(lon)

    # align and predict once
    X = feat.reindex(columns=artifacts.feature_cols)
    missing_features = [c for c in artifacts.feature_cols if c not in X.columns]
    if missing_features:
        raise ValueError(f"Missing required features after engineering: {missing_features}")

    X_imp = artifacts.imputer.transform(X)
    log_cf = artifacts.model.predict(X_imp)
    cf = np.exp(log_cf)

    pred_energy_kwh = expected_kwh * cf

    # ---- aggregate to historical monthly energies
    hourly = pd.DataFrame({"pred_energy_kwh": pred_energy_kwh}, index=ts_index)

    hist_monthly = (
        hourly["pred_energy_kwh"]
        .resample("MS")
        .sum(min_count=int(min_hours_per_month))
        .dropna()
        .to_frame("energy_kwh")
        .reset_index()
    )
    hist_monthly["year"] = hist_monthly["timestamp"].dt.year
    hist_monthly["month"] = hist_monthly["timestamp"].dt.month
    hist_monthly = hist_monthly[["year", "month", "energy_kwh"]]
    return hist_monthly


# ============================================================
# 2) Ultra-fast Monte Carlo on monthly energies
# ============================================================


def forecast_monthly_20y_from_hist_energy(
    *,
    hist_monthly: pd.DataFrame,  # columns: year, month, energy_kwh
    n_sims: int = 500,
    start_year: int = 2026,
    years: int = 20,
    random_seed: int = 42,
) -> Dict[str, object]:
    """Forecast monthly energy for the next `years` years via Monte Carlo re-sampling."""

    rng = np.random.default_rng(random_seed)

    n_months = int(years) * 12
    forecast_month_starts = pd.date_range(start=f"{int(start_year)}-01-01", periods=n_months, freq="MS", tz="UTC")
    forecast_month_numbers = forecast_month_starts.month.values

    sims = np.full((int(n_sims), n_months), np.nan, dtype=np.float64)

    # pre-split candidates by calendar month
    month_to_values = {
        m: hist_monthly.loc[hist_monthly["month"] == m, "energy_kwh"].to_numpy(dtype=float)
        for m in range(1, 13)
    }
    for m, arr in month_to_values.items():
        if arr.size == 0:
            raise ValueError(f"No historical data available for calendar month={m}")

    for s in range(int(n_sims)):
        for j in range(n_months):
            m = int(forecast_month_numbers[j])
            vals = month_to_values[m]
            sims[s, j] = vals[rng.integers(0, vals.size)]

    p10 = np.nanpercentile(sims, 10, axis=0)
    p50 = np.nanpercentile(sims, 50, axis=0)
    p90 = np.nanpercentile(sims, 90, axis=0)

    monthly_p10 = pd.DataFrame({"timestamp": forecast_month_starts, "energy_kwh": p10})
    monthly_p50 = pd.DataFrame({"timestamp": forecast_month_starts, "energy_kwh": p50})
    monthly_p90 = pd.DataFrame({"timestamp": forecast_month_starts, "energy_kwh": p90})

    totals = np.nansum(sims, axis=1)
    target_total = np.nanmedian(totals)
    rep_idx = int(np.nanargmin(np.abs(totals - target_total)))
    representative_path = pd.DataFrame({"timestamp": forecast_month_starts, "energy_kwh": sims[rep_idx, :]})

    return {
        "monthly_p10": monthly_p10,
        "monthly_p50": monthly_p50,
        "monthly_p90": monthly_p90,
        "representative_path": representative_path,
        "sims_matrix": sims,
    }
