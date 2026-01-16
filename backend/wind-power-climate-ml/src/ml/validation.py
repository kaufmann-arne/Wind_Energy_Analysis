"""Sanity checks for turbine-level ML datasets."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def validate_ml_dataset(
    df: pd.DataFrame,
    *,
    climate_prefix: str = "era5_",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Comprehensive sanity checks for a turbine-level ML dataset.

    Assumes the format produced by `src.ml.dataset_builder.build_ml_datasets_per_turbine`.
    Returns a small report dictionary.
    """

    if verbose:
        print("=" * 80)
        print("ML DATASET VALIDATION")
        print("=" * 80)

    required_columns = [
        "timestamp",
        "turbine_id",
        "site_id",
        "energy_kwh",
        "expected_energy_kwh",
        "target_correction_factor",
        "target_is_valid",
        "hub_height_m",
        "rated_power_kw",
        "turbine_type",
    ]

    climate_required = [
        f"{climate_prefix}ws_hub",
        f"{climate_prefix}ws_10m",
        f"{climate_prefix}ws_100m",
        f"{climate_prefix}shear_alpha",
        f"{climate_prefix}air_density_kgm3",
    ]

    missing = set(required_columns + climate_required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if verbose:
        print("✓ All required columns present")

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("NaT found in timestamp column")

    if ts.dt.tz is None:
        raise ValueError("Timestamp is timezone-naive (must be UTC-aware)")
    if verbose:
        print(f"✓ Timestamp timezone: {ts.dt.tz}")

    deltas = ts.sort_values().diff().dropna()
    median_delta = deltas.median()
    if verbose:
        print(f"✓ Median timestep: {median_delta}")
        if not pd.Timedelta("50min") <= median_delta <= pd.Timedelta("70min"):
            print("⚠️ WARNING: Dataset does not appear to be hourly")
        print(f"✓ Time span: {ts.min()} → {ts.max()}")

    if (df["energy_kwh"] < 0).any():
        raise ValueError("Negative energy_kwh values found")
    if (df["expected_energy_kwh"] < 0).any():
        raise ValueError("Negative expected_energy_kwh values found")

    rated_power = float(df["rated_power_kw"].iloc[0])
    if (df["energy_kwh"] > rated_power * 1.05).any() and verbose:
        print("⚠️ WARNING: Observed energy exceeds rated power by >5%")
    if verbose:
        print("✓ Energy values plausible")

    ws = df[f"{climate_prefix}ws_hub"]
    if (ws < 0).any():
        raise ValueError("Negative wind speeds detected")
    if ws.max() > 40 and verbose:
        print("⚠️ WARNING: Very high wind speeds (>40 m/s) detected")
    if verbose:
        print("✓ Wind speed values plausible")

    cf = df["target_correction_factor"]
    invalid_cf = cf[~cf.isna() & ~np.isfinite(cf)]
    if not invalid_cf.empty:
        raise ValueError("Non-finite correction factors detected")
    if (cf < 0).any():
        raise ValueError("Negative correction factors detected")

    if verbose:
        print(
            "✓ Correction factor stats:",
            f"mean={cf.mean():.3f}",
            f"std={cf.std():.3f}",
            f"min={cf.min():.3f}",
            f"max={cf.max():.3f}",
        )

    valid_frac = float(df["target_is_valid"].mean())
    if verbose:
        print(f"✓ target_is_valid fraction: {100 * valid_frac:.1f}%")
        if valid_frac < 0.2:
            print("⚠️ WARNING: Very few valid training points")
        if valid_frac > 0.9:
            print("⚠️ WARNING: Almost all points are valid – check filters")

    if "target_log_correction_factor" in df.columns:
        log_cf = df["target_log_correction_factor"]
        if (~log_cf.isna() & ~np.isfinite(log_cf)).any():
            raise ValueError("Non-finite log correction factors detected")
        if verbose:
            print(
                "✓ Log-target stats:",
                f"mean={log_cf.mean():.3f}",
                f"std={log_cf.std():.3f}",
            )

    missing_era5 = df[climate_required].isna().mean()
    if verbose:
        print("✓ ERA5 missing fractions:")
        for k, v in missing_era5.items():
            print(f"  {k}: {100 * v:.2f}%")
        if (missing_era5 > 0.05).any():
            print("⚠️ WARNING: Significant ERA5 merge gaps detected")

        print("=" * 80)
        print("VALIDATION COMPLETED")
        print("=" * 80)

    return {
        "rows": int(len(df)),
        "valid_fraction": valid_frac,
        "time_start": ts.min(),
        "time_end": ts.max(),
    }
