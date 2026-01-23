#!/usr/bin/env python
"""CLI wrapper for ERA5 downloads.

Examples
--------
# Germany bounding box
python scripts/download_era5.py germany \
  --outdir "Data/Raw/Era5/ERA5_Germany" \
  --start-year 2004 --end-year 2024

# Point/buffer monthly
python scripts/download_era5.py point \
  --lat 55.679306 --lon 11.274972 --buffer-deg 0.25 \
  --start-date 2019-12-01 --end-date 2025-11-24 \
  --outdir "Data/Raw/Era5/ERA5_Loegtved_fsr_zust" \
  --variables forecast_surface_roughness friction_velocity
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.era5.download import (
    download_era5_germany_bbox,
    download_era5_monthly_for_area,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_de = sub.add_parser("germany", help="Download ERA5 for Germany bbox")
    p_de.add_argument("--outdir", required=True)
    p_de.add_argument("--start-year", type=int, required=True)
    p_de.add_argument("--end-year", type=int, required=True)

    p_pt = sub.add_parser("point", help="Download ERA5 for a small area around a point")
    p_pt.add_argument("--lat", type=float, required=True)
    p_pt.add_argument("--lon", type=float, required=True)
    p_pt.add_argument("--buffer-deg", type=float, default=0.25)
    p_pt.add_argument("--start-date", required=True)
    p_pt.add_argument("--end-date", required=True)
    p_pt.add_argument("--outdir", required=True)
    p_pt.add_argument("--variables", nargs="+", required=True)

    args = parser.parse_args()

    if args.mode == "germany":
        download_era5_germany_bbox(
            outdir=Path(args.outdir),
            start_year=args.start_year,
            end_year=args.end_year,
        )
        return

    if args.mode == "point":
        area = [
            args.lat + args.buffer_deg,
            args.lon - args.buffer_deg,
            args.lat - args.buffer_deg,
            args.lon + args.buffer_deg,
        ]
        download_era5_monthly_for_area(
            outdir=Path(args.outdir),
            area=area,
            start_date=args.start_date,
            end_date=args.end_date,
            variables=args.variables,
        )
        return


if __name__ == "__main__":
    main()
