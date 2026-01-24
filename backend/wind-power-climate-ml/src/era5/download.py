"""
ERA5 download utilities via CDS API.

This module provides project-compatible, reproducible ERA5 downloads with:
- Monthly NetCDF outputs (one file per year-month)
- Hourly timesteps (00:00..23:00)
- Day list 01..31 (CDS ignores invalid days per month)
- Bounding boxes defined as [North, West, South, East]

It includes convenience wrappers used by scripts/download_era5.py:
- download_era5_germany_bbox(...): Germany bbox with custom filename template
- download_era5_monthly_for_area(...): monthly loop for arbitrary area and date range

Prerequisites
-------------
- A configured CDS API key in ~/.cdsapirc
- `cdsapi` installed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

try:
    import cdsapi
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cdsapi is required for ERA5 downloads. Install it via `pip install cdsapi`."
    ) from e


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
HOURS_UTC: List[str] = [f"{h:02d}:00" for h in range(24)]
DAYS_01_31: List[str] = [f"{d:02d}" for d in range(1, 32)]
MONTHS_01_12: List[str] = [f"{m:02d}" for m in range(1, 13)]


# ---------------------------------------------------------------------
# Types / helpers
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class BoundingBox:
    """ERA5 bounding box: [N, W, S, E] in degrees."""

    north: float
    west: float
    south: float
    east: float

    def as_list(self) -> List[float]:
        return [self.north, self.west, self.south, self.east]


AreaLike = Union[BoundingBox, Sequence[float]]  # [N, W, S, E]


def _as_bbox(area: AreaLike) -> BoundingBox:
    """Accept BoundingBox or [N, W, S, E] list/tuple."""
    if isinstance(area, BoundingBox):
        return area
    if not (isinstance(area, (list, tuple)) and len(area) == 4):
        raise ValueError("area must be BoundingBox or a 4-element sequence [N, W, S, E]")
    return BoundingBox(
        north=float(area[0]),
        west=float(area[1]),
        south=float(area[2]),
        east=float(area[3]),
    )


# ---------------------------------------------------------------------
# Project presets
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Core download functions
# ---------------------------------------------------------------------
def download_era5_monthly(
    *,
    outdir: Path,
    start_year: int,
    end_year: int,
    area: AreaLike,
    variables: Sequence[str],
    dataset: str = "reanalysis-era5-single-levels",
    product_type: str = "reanalysis",
    file_format: str = "netcdf",
    hours: Sequence[str] = HOURS_UTC,
    days: Sequence[str] = DAYS_01_31,
    months: Sequence[str] = MONTHS_01_12,
    filename_template: str = "era5_{tag}_{year}_{month}.nc",
    tag: str = "area",
    client: Optional["cdsapi.Client"] = None,
) -> None:
    """Download ERA5 single-levels for a bounding box, one NetCDF per month.

    Parameters
    ----------
    outdir:
        Target output directory.
    start_year, end_year:
        Inclusive range.
    area:
        BoundingBox or [N, W, S, E]
    variables:
        ERA5 variable names.
    filename_template:
        May include {tag}, {year}, {month}. (month already zero-padded)
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bbox = _as_bbox(area)
    c = client or cdsapi.Client()

    for year in range(int(start_year), int(end_year) + 1):
        for month in months:
            out_file = outdir / filename_template.format(tag=tag, year=year, month=month)

            if out_file.exists():
                print(f"✓ Already there: {out_file.name}")
                continue

            print(f"↓ Download ERA5: {year}-{month} → {out_file}")

            c.retrieve(
                dataset,
                {
                    "product_type": product_type,
                    "format": file_format,
                    "variable": list(variables),
                    "year": str(year),
                    "month": str(month),
                    "day": list(days),
                    "time": list(hours),
                    "area": bbox.as_list(),
                },
                str(out_file),
            )


def download_era5_monthly_for_area(
    *,
    outdir: Path,
    area: Sequence[float],  # [N, W, S, E]
    start_date: str,
    end_date: str,
    variables: Sequence[str],
    dataset: str = "reanalysis-era5-single-levels",
    client: Optional["cdsapi.Client"] = None,
    filename_template: str = "era5_{year}_{month}.nc",
    file_format: str = "netcdf",
    hours: Sequence[str] = HOURS_UTC,
    days: Sequence[str] = DAYS_01_31,
) -> None:
    """Monthly loop between start_date and end_date (month starts), one NetCDF per month.

    Dates are interpreted as:
    - start_date inclusive
    - end_date inclusive, but internally we request months via MS (month starts).

    filename_template may use {year} and {month}.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date, freq="MS")
    c = client or cdsapi.Client()

    for d in dates:
        year = int(d.year)
        month = f"{int(d.month):02d}"
        outfile = outdir / filename_template.format(year=year, month=month)

        if outfile.exists():
            print(f"✓ Already there: {outfile.name}")
            continue

        print(f"↓ Downloading ERA5 {year}-{month} → {outfile}")

        c.retrieve(
            dataset,
            {
                "product_type": "reanalysis",
                "format": file_format,
                "variable": list(variables),
                "year": str(year),
                "month": month,
                "day": list(days),
                "time": list(hours),
                "area": list(area),
            },
            str(outfile),
        )


# ---------------------------------------------------------------------
# Convenience wrappers used by scripts
# ---------------------------------------------------------------------
def download_era5_germany_bbox(
    *,
    outdir: Path,
    start_year: int,
    end_year: int,
    variables: Sequence[str] = GERMANY_DEFAULT_VARIABLES,
    dataset: str = "reanalysis-era5-single-levels",
    client: Optional["cdsapi.Client"] = None,
    filename_template: str = "era5_de_{year}_{month}.nc",
) -> None:
    """Germany bbox, monthly NetCDFs with custom naming."""
    download_era5_monthly(
        outdir=outdir,
        start_year=start_year,
        end_year=end_year,
        area=GERMANY_BBOX,
        variables=variables,
        dataset=dataset,
        filename_template=filename_template,
        tag="de",
        client=client,
    )

