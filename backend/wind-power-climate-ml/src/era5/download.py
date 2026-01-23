"""ERA5 download utilities via CDS API.

This module is a parameterized version of the notebook snippets you provided.
It keeps the same practical behavior:

- Monthly NetCDF outputs (one file per year-month)
- Hourly timesteps (00:00..23:00)
- Day list 01..31 (CDS will ignore invalid days per month)
- Bounding box defined as [North, West, South, East]

Prerequisites
-------------
- A configured CDS API key in ~/.cdsapirc
- `cdsapi` installed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

try:
    import cdsapi
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cdsapi is required for ERA5 downloads. Install it via `pip install cdsapi`."
    ) from e


HOURS_UTC: List[str] = [f"{h:02d}:00" for h in range(24)]
DAYS_01_31: List[str] = [f"{d:02d}" for d in range(1, 32)]
MONTHS_01_12: List[str] = [f"{m:02d}" for m in range(1, 13)]


@dataclass(frozen=True)
class BoundingBox:
    """ERA5 bounding box: [N, W, S, E] in degrees."""

    north: float
    west: float
    south: float
    east: float

    def as_list(self) -> List[float]:
        return [self.north, self.west, self.south, self.east]


def download_era5_monthly(
    *,
    outdir: Path,
    start_year: int,
    end_year: int,
    area: BoundingBox,
    variables: Sequence[str],
    dataset: str = "reanalysis-era5-single-levels",
    product_type: str = "reanalysis",
    data_format: str = "netcdf",
    hours: Sequence[str] = HOURS_UTC,
    days: Sequence[str] = DAYS_01_31,
    months: Sequence[str] = MONTHS_01_12,
    filename_template: str = "era5_{tag}_{year}_{month}.nc",
    tag: str = "",
    client: Optional["cdsapi.Client"] = None,
) -> None:
    """Download ERA5 single-levels for a bounding box, one NetCDF per month."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    c = client or cdsapi.Client()

    for year in range(int(start_year), int(end_year) + 1):
        for month in months:
            out_file = outdir / filename_template.format(tag=tag or "area", year=year, month=month)

            if out_file.exists():
                print(f"✓ Already there: {out_file.name}")
                continue

            print(f"↓ Download ERA5: {year}-{month} → {out_file.name}")

            c.retrieve(
                dataset,
                {
                    "product_type": product_type,
                    "data_format": data_format,
                    "variable": list(variables),
                    "year": str(year),
                    "month": str(month),
                    "day": list(days),
                    "time": list(hours),
                    "area": area.as_list(),
                },
                str(out_file),
            )


def download_era5_point_monthly(
    *,
    lat_center: float,
    lon_center: float,
    start_date: str,
    end_date: str,
    outdir: Path,
    variables: Sequence[str],
    area_buffer_deg: float = 0.25,
    dataset: str = "reanalysis-era5-single-levels",
    filename_template: str = "era5_{year}_{month}.nc",
    client: Optional["cdsapi.Client"] = None,
) -> None:
    """Download ERA5 for a lat/lon center using a small buffer box (e.g., 2x2 grid points).

    This mirrors your "EFFIZIENT & PROJEKTKOMPATIBEL" monthly loop.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date, freq="MS")

    area = BoundingBox(
        north=lat_center + area_buffer_deg,
        west=lon_center - area_buffer_deg,
        south=lat_center - area_buffer_deg,
        east=lon_center + area_buffer_deg,
    )

    c = client or cdsapi.Client()

    for d in dates:
        year = int(d.year)
        month = f"{int(d.month):02d}"

        outfile = outdir / filename_template.format(year=year, month=month)
        if outfile.exists():
            print(f"✓ ERA5 {year}-{month} already exists")
            continue

        print(f"↓ Downloading ERA5 {year}-{month} ...")

        c.retrieve(
            dataset,
            {
                "product_type": "reanalysis",
                "variable": list(variables),
                "year": str(year),
                "month": month,
                "day": DAYS_01_31,
                "time": HOURS_UTC,
                "area": area.as_list(),
                "format": "netcdf",
            },
            str(outfile),
        )


# Convenience presets matching your notebooks
GERMANY_BBOX = BoundingBox(north=56.0, west=5.1, south=46.5, east=16.0)

GERMANY_DEFAULT_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "forecast_surface_roughness",
    "friction_velocity",
    "2m_dewpoint_temperature",
]
