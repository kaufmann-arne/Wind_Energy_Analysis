"""Run the website-style 20-year monthly production forecast.

This script is designed for local testing without requiring an editable install.
It appends `<repo_root>/src` to `sys.path`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from website.forecast import (
    build_hist_monthly_energy_lookup,
    forecast_monthly_20y_from_hist_energy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--hub-height", type=float, required=True)
    p.add_argument("--turbine-type", type=str, required=True)

    p.add_argument("--era5-main-dir", type=str, required=True)
    p.add_argument("--era5-d2m-dir", type=str, required=True)

    p.add_argument(
        "--turbine-power-curve-json",
        type=str,
        default=str(REPO_ROOT / "data" / "turbine_power_curves.json"),
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=str(REPO_ROOT / "LGBM_Model" / "model_artifacts"),
    )

    p.add_argument("--start-year", type=int, default=2026)
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--n-sims", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "website_forecast"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = build_hist_monthly_energy_lookup(
        lat=args.lat,
        lon=args.lon,
        hub_height_m=args.hub_height,
        turbine_type=args.turbine_type,
        era5_main_dir=Path(args.era5_main_dir),
        era5_d2m_dir=Path(args.era5_d2m_dir),
        turbine_power_curve_json=Path(args.turbine_power_curve_json),
        model_dir=Path(args.model_dir),
    )

    fc = forecast_monthly_20y_from_hist_energy(
        hist_monthly=hist,
        n_sims=args.n_sims,
        start_year=args.start_year,
        years=args.years,
        random_seed=args.seed,
    )

    hist.to_csv(out_dir / "hist_monthly_energy_kwh.csv", index=False)
    fc["monthly_p10"].to_csv(out_dir / "forecast_monthly_p10_kwh.csv", index=False)
    fc["monthly_p50"].to_csv(out_dir / "forecast_monthly_p50_kwh.csv", index=False)
    fc["monthly_p90"].to_csv(out_dir / "forecast_monthly_p90_kwh.csv", index=False)
    fc["representative_path"].to_csv(out_dir / "forecast_representative_path_kwh.csv", index=False)

    print("Saved outputs to:", out_dir)
    print("P50 head:")
    print(fc["monthly_p50"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
