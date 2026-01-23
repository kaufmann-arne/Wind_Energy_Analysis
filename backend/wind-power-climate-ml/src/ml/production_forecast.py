"""
Production-time (website) inference and long-horizon monthly forecasting.

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

from functools import lru_cache
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import re
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from time import perf_counter


# ============================================================
# Helpers (must match training logic)
# ============================================================

@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    imputer: object
    feature_cols: list[str]
    baseline_monthly_logcf: np.ndarray
    baseline_global_logcf: float


def load_model_artifacts(model_dir: Path) -> ModelArtifacts:
    model = joblib.load(Path(model_dir) / "final_model.pkl")
    imputer = joblib.load(Path(model_dir) / "feature_imputer.pkl")
    with open(Path(model_dir) / "feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    baseline_path = Path(model_dir) / "baseline_monthly_logcf.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        monthly = np.asarray(baseline["baseline_monthly_logcf"], dtype=float)
        if monthly.shape != (12,):
            raise ValueError("baseline_monthly_logcf must have length 12")
        global_logcf = float(baseline.get("baseline_global_logcf", 0.0))
    else:
        # fallback: "no correction" baseline
        monthly = np.zeros(12, dtype=float)
        global_logcf = 0.0

    return ModelArtifacts(
        model=model,
        imputer=imputer,
        feature_cols=feature_cols,
        baseline_monthly_logcf=monthly,
        baseline_global_logcf=global_logcf,
    )


@lru_cache(maxsize=4)
def _cached_artifacts(model_dir_str: str) -> ModelArtifacts:
    return load_model_artifacts(Path(model_dir_str))


@lru_cache(maxsize=8)
def _cached_curves(curve_json_path_str: str) -> dict:
    p = Path(curve_json_path_str)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_turbine_by_id(curves: dict, turbine_type_id: int) -> Tuple[str, dict]:
    """
    Resolve turbine by numeric ID stored in each turbine meta (meta["id"]).
    Returns (turbine_key, meta).
    """
    tid = int(turbine_type_id)
    for key, meta in curves.items():
        if "id" in meta and int(meta["id"]) == tid:
            return key, meta
    raise KeyError(f"Turbine id not found in turbine_power_curve_json: {tid}")


@lru_cache(maxsize=4)
def _cached_era5_dataset(
    era5_dir_str: str,
    engine: str | None,
    sample_pct: int,
    random_seed: int,
    min_files_per_month: int,
) -> xr.Dataset:
    return _open_era5_dir(
        Path(era5_dir_str),
        engine=engine,
        sample_frac=sample_pct / 100.0,
        random_seed=random_seed,
        min_files_per_month=min_files_per_month,
    )


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


def expected_power_kw_from_curve(
    ws: np.ndarray,
    curve_ws: np.ndarray,
    curve_p_kw: np.ndarray,
    *,
    cut_out_ws: float | None = None,
) -> np.ndarray:
    ws = np.asarray(ws, dtype=float)

    p = np.interp(
        ws,
        curve_ws,
        curve_p_kw,
        left=0.0,
        right=float(np.max(curve_p_kw)),
    )

    if cut_out_ws is not None and np.isfinite(cut_out_ws):
        p = np.where(ws >= float(cut_out_ws), 0.0, p)

    return p


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


def _open_era5_dir(
    nc_dir: Path,
    *,
    engine: Optional[str] = None,
    sample_frac: float = 1.0,
    random_seed: int = 42,
    min_files_per_month: int = 3,
) -> xr.Dataset:
    files = sorted(Path(nc_dir).glob("*.nc"))
    if not files:
        raise ValueError(f"No .nc files found in: {nc_dir}")

    # --- optional subsampling of MONTHLY files, stratified by calendar month ---
    if sample_frac < 1.0:
        rng = np.random.default_rng(int(random_seed))

        # expects filenames like ..._YYYY_MM.nc (e.g., era5_de_2009_11.nc)
        month_to_files = {m: [] for m in range(1, 13)}
        unmatched = []

        pat = re.compile(r".*_(\d{4})_(\d{2})\.nc$")

        for f in files:
            m = pat.match(f.name)
            if not m:
                unmatched.append(f)
                continue
            month = int(m.group(2))
            if 1 <= month <= 12:
                month_to_files[month].append(f)
            else:
                unmatched.append(f)

        # if parsing fails for many files, fall back to full set (safety)
        parsed_count = sum(len(v) for v in month_to_files.values())
        if parsed_count >= 12:  # at least some coverage
            sampled = []
            for month, flist in month_to_files.items():
                if not flist:
                    continue
                k = max(min_files_per_month, int(np.ceil(sample_frac * len(flist))))
                k = min(k, len(flist))
                idx = rng.choice(len(flist), size=k, replace=False)
                sampled.extend([flist[i] for i in idx])

            # keep unmatched files only in full mode (they might be non-monthly or odd)
            files = sorted(sampled)
        # else: silently keep full files

    kwargs = dict(combine="by_coords")
    if engine is not None:
        kwargs["engine"] = engine

    ds = xr.open_mfdataset(files, **kwargs)
    if "valid_time" in ds:
        ds = ds.rename({"valid_time": "time"})
    return ds


# ============================================================
# 1) Historical monthly energies (ML-corrected)
# ============================================================

def build_hist_monthly_energy_lookup(
    *,
    lat: float,
    lon: float,
    hub_height_m: float,
    turbine_type_id: int,
    era5_main_dir: Path,
    turbine_power_curve_json: Path,
    model_dir: Path,
    era5_sample_frac: float = 1.0,
    random_seed: int = 42,
    blend_weight_w: float = 0.9,
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
    t0 = perf_counter()
    artifacts = _cached_artifacts(str(Path(model_dir).resolve()))

    # ---- power curve + turbine meta (RESOLVE BY ID)
    curves = _cached_curves(str(Path(turbine_power_curve_json).resolve()))
    turbine_key, meta = _resolve_turbine_by_id(curves, int(turbine_type_id))

    curve = np.asarray(meta["power_curve"], dtype=float)
    cut_out_ws = float(meta.get("Cut-out wind speed", np.inf))
    curve_ws = curve[:, 0]
    curve_p_kw = curve[:, 1]

    rated_power_kw = float(meta["rated_power_kw"])
    rotor_diameter_m = float(meta["rotor_diameter_m"])
    area = np.pi * (rotor_diameter_m / 2.0) ** 2
    specific_power_wpm2 = (rated_power_kw * 1000.0) / area

    # ---- load ERA5
    t_load0 = perf_counter()
    sample_pct = int(round(float(era5_sample_frac) * 100))
    ds = _cached_era5_dataset(
        str(Path(era5_main_dir).resolve()),
        xarray_engine,
        sample_pct,
        int(random_seed),
        3,  # min_files_per_month
    )
    t_load1 = perf_counter()

    t_interp0 = perf_counter()
    ds_i = ds.interp(latitude=float(lat), longitude=float(lon))
    t_interp1 = perf_counter()

    # ---- time index (take from interpolated dataset BEFORE subsetting)
    ts = pd.to_datetime(ds_i["time"].values, utc=True)
    ts_index = pd.DatetimeIndex(ts, name="timestamp")
    tf = cyclical_time_features(ts_index)

    # ---- materialize needed vars ONCE (prevents repeated expensive .values reads)
    vars_needed = ["u10", "v10", "u100", "v100", "t2m", "sp", "d2m", "fsr", "zust"]
    vars_needed = [v for v in vars_needed if v in ds_i]

    t_mat0 = perf_counter()
    ds_i = ds_i[vars_needed].load()
    t_mat1 = perf_counter()

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

    expected_kw = expected_power_kw_from_curve(ws_hub, curve_ws, curve_p_kw, cut_out_ws=cut_out_ws)
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

    # align and predict once
    t_pred0 = perf_counter()
    X = feat.reindex(columns=artifacts.feature_cols)
    missing_features = [c for c in artifacts.feature_cols if c not in X.columns]
    if missing_features:
        raise ValueError(f"Missing required features after engineering: {missing_features}")

    X_imp = artifacts.imputer.transform(X)
    X_imp = pd.DataFrame(X_imp, columns=artifacts.feature_cols, index=X.index)

    log_cf_ml = artifacts.model.predict(X_imp)

    w = float(blend_weight_w)
    w = max(0.0, min(1.0, w))

    if w < 1.0:
        months = ts_index.month.values  # 1..12
        log_cf_base = artifacts.baseline_monthly_logcf[months - 1]
        log_cf = (1.0 - w) * log_cf_base + w * log_cf_ml
    else:
        log_cf = log_cf_ml

    LOGCF_MIN = -3.9
    LOGCF_MAX = 2.30
    log_cf = np.clip(log_cf, LOGCF_MIN, LOGCF_MAX)

    cf = np.exp(log_cf)
    pred_energy_kwh = expected_kwh * cf

    # Optional physical cap per hour
    pred_energy_kwh = np.minimum(pred_energy_kwh, rated_power_kw * 1.05)
    t_pred1 = perf_counter()

    # ---- aggregate to historical monthly energies
    hourly = pd.DataFrame({"pred_energy_kwh": pred_energy_kwh}, index=ts_index)
    t_res0 = perf_counter()
    hist_monthly = (
        hourly["pred_energy_kwh"]
        .resample("MS")
        .sum(min_count=int(min_hours_per_month))
        .dropna()
        .to_frame("energy_kwh")
        .reset_index()
    )
    if "timestamp" not in hist_monthly.columns:
        hist_monthly = hist_monthly.rename(columns={hist_monthly.columns[0]: "timestamp"})
    t_res1 = perf_counter()
    t1 = perf_counter()

    print(
        f"[HIST_TIMING] turbine_id={int(turbine_type_id)} key={turbine_key} | "
        f"total={t1-t0:.3f}s | era5_open={t_load1-t_load0:.3f}s | "
        f"interp={t_interp1-t_interp0:.3f}s | materialize={t_mat1-t_mat0:.3f}s | "
        f"predict={t_pred1-t_pred0:.3f}s | resample={t_res1-t_res0:.3f}s"
    )

    hist_monthly["year"] = hist_monthly["timestamp"].dt.year
    hist_monthly["month"] = hist_monthly["timestamp"].dt.month
    hist_monthly = hist_monthly[["year", "month", "energy_kwh"]]
    return hist_monthly


# ============================================================
# 2) Ultra-fast Monte Carlo on monthly energies
# ============================================================

def forecast_monthly_representative_path_from_hist_energy(
    *,
    hist_monthly: pd.DataFrame,          # columns: year, month, energy_kwh
    start_date: str | pd.Timestamp,      # COD-like date, uses year+month
    years: int = 20,
    n_sims: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Returns ONE monthly path (representative simulation):
    choose the simulated path whose TOTAL energy is closest to the median total.

    start_date:
      Any date-like string / Timestamp. We normalize to month start (MS) in UTC,
      so COD "2026-03-17" becomes "2026-03-01".
    """
    rng = np.random.default_rng(int(random_seed))

    n_months = int(years) * 12

    start_ts = pd.Timestamp(start_date)
    start_ts = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC")

    month_starts = pd.date_range(
        start=start_ts,
        periods=n_months,
        freq="MS",
        tz="UTC",
    )
    month_nums = month_starts.month.values

    # split historical candidates by calendar month
    month_to_values = {
        m: hist_monthly.loc[hist_monthly["month"] == m, "energy_kwh"].to_numpy(dtype=float)
        for m in range(1, 13)
    }
    for m, arr in month_to_values.items():
        if arr.size == 0:
            raise ValueError(f"No historical data available for calendar month={m}")

    # simulate
    sims = np.empty((int(n_sims), n_months), dtype=np.float64)
    for s in range(int(n_sims)):
        for j in range(n_months):
            vals = month_to_values[int(month_nums[j])]
            sims[s, j] = vals[rng.integers(0, vals.size)]

    # pick representative path: total closest to median total
    totals = np.nansum(sims, axis=1)
    target_total = np.nanmedian(totals)
    rep_idx = int(np.nanargmin(np.abs(totals - target_total)))

    return pd.DataFrame(
        {"timestamp": month_starts, "energy_kwh": sims[rep_idx, :]},
    )
