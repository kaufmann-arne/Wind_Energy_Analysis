#!/usr/bin/env python
"""CLI wrapper for building ML datasets (SCADA + ERA5 merge)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.dataset_builder import build_ml_datasets_per_turbine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--era5-main-dir", type=Path, required=True)
    p.add_argument("--era5-fsr-zust-dir", type=Path, required=True)
    p.add_argument("--scada-csv", type=Path, required=True)
    p.add_argument("--turbine-curve-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--climate-prefix", type=str, default="era5_")
    p.add_argument("--target-tz", type=str, default="UTC")
    p.add_argument("--merge-tolerance", type=str, default="45min")
    p.add_argument("--no-log-target", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = build_ml_datasets_per_turbine(
        era5_main_paths=list(args.era5_main_dir.glob("*.nc")),
        era5_fsr_zust_paths=list(args.era5_fsr_zust_dir.glob("*.nc")),
        scada_csv_path=args.scada_csv,
        turbine_power_curve_json=args.turbine_curve_json,
        output_dir=args.output_dir,
        climate_prefix=args.climate_prefix,
        target_tz=args.target_tz,
        merge_tolerance=args.merge_tolerance,
        use_log_target=not args.no_log_target,
    )
    print("Built datasets:", ", ".join(datasets.keys()))


if __name__ == "__main__":
    main()
