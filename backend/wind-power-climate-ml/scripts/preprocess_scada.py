#!/usr/bin/env python
"""CLI wrapper around `src.scada.preprocess.preprocess_scada_dataset`.

This is intentionally thin: for real usage, put per-site configs in `config/` and
call this script from your pipeline runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.scada.preprocess import preprocess_scada_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="SCADA directory or single CSV")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--site-id", required=True)
    p.add_argument("--site-lat", type=float, required=True)
    p.add_argument("--site-lon", type=float, required=True)
    p.add_argument("--timestamp-col", required=True)
    p.add_argument("--target-col", required=True)
    p.add_argument("--windspeed-col", required=True)
    p.add_argument("--turbine-db", required=True)
    p.add_argument("--turbine-meta-json", required=True, help="Path to turbine_meta JSON")
    p.add_argument("--turbine-col", default=None)
    p.add_argument("--qc-col", default=None)
    p.add_argument("--min-power", type=float, default=0.0)
    p.add_argument("--max-power", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    turbine_meta_path = Path(args.turbine_meta_json)
    with turbine_meta_path.open("r", encoding="utf-8") as f:
        turbine_meta = json.load(f)

    preprocess_scada_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        site_id=args.site_id,
        site_lat=args.site_lat,
        site_lon=args.site_lon,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        windspeed_col=args.windspeed_col,
        turbine_col=args.turbine_col,
        turbine_meta=turbine_meta,
        turbine_db_path=Path(args.turbine_db),
        qc_col=args.qc_col,
        min_power=args.min_power,
        max_power=args.max_power,
    )


if __name__ == "__main__":
    main()
