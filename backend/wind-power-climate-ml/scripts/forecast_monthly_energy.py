"""Build historical monthly energy (ML-corrected) and forecast 20 years.

This is a CLI wrapper around :mod:`ml.production_forecast` and is intended for
the website backend / batch jobs.

Example:
  python scripts/forecast_monthly_energy.py \
    --lat 52.52 --lon 13.41 --hub-height-m 75 \
    --turbine-type Senvion_MM82 \
    --era5-main-dir Data/Raw/Era5/ERA5_Germany_ClimateStore \
    --era5-d2m-dir  Data/Raw/Era5/ERA5_Germany_d2m \
    --turbine-curves Data/turbine_power_curves.json \
    --model-dir artifacts/LGBM_Model \
    --out-dir artifacts/forecast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ml.production_forecast import (
    build_hist_monthly_energy_lookup,
    forecast_monthly_20y_from_hist_energy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--hub-height-m", type=float, required=True)
    p.add_argument("--turbine-type", type=str, required=True)
    p.add_argument("--era5-main-dir", type=str, required=True)
    p.add_argument("--era5-d2m-dir", type=str, required=True)
    p.add_argument("--turbine-curves", type=str, required=True)
    p.add_argument("--model-dir", type=str, required=True)

    p.add_argument("--out-dir", type=str, default="artifacts/forecast")
    p.add_argument("--start-year", type=int, default=2026)
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--n-sims", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = build_hist_monthly_energy_lookup(
        lat=a.lat,
        lon=a.lon,
        hub_height_m=a.hub_height_m,
        turbine_type=a.turbine_type,
        era5_main_dir=Path(a.era5_main_dir),
        era5_d2m_dir=Path(a.era5_d2m_dir),
        turbine_power_curve_json=Path(a.turbine_curves),
        model_dir=Path(a.model_dir),
    )

    out = forecast_monthly_20y_from_hist_energy(
        hist_monthly=hist,
        n_sims=a.n_sims,
        start_year=a.start_year,
        years=a.years,
        random_seed=a.seed,
    )

    # persist outputs
    hist.to_csv(out_dir / "hist_monthly_energy.csv", index=False)
    out["monthly_p10"].to_csv(out_dir / "forecast_monthly_p10.csv", index=False)
    out["monthly_p50"].to_csv(out_dir / "forecast_monthly_p50.csv", index=False)
    out["monthly_p90"].to_csv(out_dir / "forecast_monthly_p90.csv", index=False)
    out["representative_path"].to_csv(out_dir / "forecast_representative_path.csv", index=False)

    print("Saved:")
    for f in [
        out_dir / "hist_monthly_energy.csv",
        out_dir / "forecast_monthly_p10.csv",
        out_dir / "forecast_monthly_p50.csv",
        out_dir / "forecast_monthly_p90.csv",
        out_dir / "forecast_representative_path.csv",
    ]:
        print(" -", f)


if __name__ == "__main__":
    main()
