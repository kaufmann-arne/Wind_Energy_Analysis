#!/usr/bin/env python
"""
Project-default ERA5 downloader.

Runs without requiring any CLI inputs:
    python scripts/download_era5.py

Reads config from:
    config/sites.yaml  (default)

Behavior:
- Germany bbox: all variables in one monthly file
    data/raw/era5/ERA5_Germany/era5_de_YYYY_MM.nc

- Point sites: two separate monthly downloads per site
  Core variables:
    data/raw/era5/ERA5_<site>/era5_YYYY_MM.nc
  Roughness variables (fsr + zust):
    data/raw/era5/ERA5_<site>_fsr_zust/era5_YYYY_MM.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.era5.download import (
    download_era5_germany_bbox,
    download_era5_monthly_for_area,
)

try:
    import yaml
except ImportError:  
    yaml = None


def repo_root() -> Path:
    # scripts/ is directly under repo root
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "Missing dependency: pyyaml. Install via `pip install pyyaml` "
            "and add it to requirements.txt/pyproject.toml."
        )
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def require_key(d: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {where}")
    return d[key]


def run_defaults(config_path: Path) -> None:
    cfg = load_yaml(config_path)

    era5 = cfg.get("era5", {})
    if not isinstance(era5, dict) or not era5:
        raise KeyError("Missing 'era5' block in config/sites.yaml")

    # ---------------- Germany: all variables in one monthly file ----------------
    germany = require_key(era5, "germany", "era5")
    outdir_de = repo_root() / Path(require_key(germany, "outdir", "era5.germany"))
    start_year = int(require_key(germany, "start_year", "era5.germany"))
    end_year = int(require_key(germany, "end_year", "era5.germany"))
    variables_all = require_key(germany, "variables_all", "era5.germany")

    download_era5_germany_bbox(
        outdir=outdir_de,
        start_year=start_year,
        end_year=end_year,
        variables=variables_all,
        filename_template="era5_de_{year}_{month}.nc",
    )

    # ---------------- Points: core vs roughness in separate folders ----------------
    points = require_key(era5, "points", "era5")
    outdir_root = repo_root() / Path(require_key(points, "outdir_root", "era5.points"))
    buffer_deg_default = float(points.get("buffer_deg", 0.25))

    core_vars = require_key(points, "core_variables", "era5.points")
    rough_vars = require_key(points, "roughness_variables", "era5.points")
    sites = require_key(points, "sites", "era5.points")

    if not isinstance(sites, dict) or not sites:
        print("No point sites configured under era5.points.sites -> skipping point downloads.")
        return

    for site_name, s in sites.items():
        if not isinstance(s, dict):
            raise TypeError(f"era5.points.sites.{site_name} must be a mapping")

        lat = float(require_key(s, "lat", f"era5.points.sites.{site_name}"))
        lon = float(require_key(s, "lon", f"era5.points.sites.{site_name}"))

        # per-site dates are required for your project
        start_date = str(require_key(s, "start_date", f"era5.points.sites.{site_name}"))
        end_date = str(require_key(s, "end_date", f"era5.points.sites.{site_name}"))

        buffer_deg = float(s.get("buffer_deg", buffer_deg_default))
        area = [lat + buffer_deg, lon - buffer_deg, lat - buffer_deg, lon + buffer_deg]

        # Core folder: data/raw/era5/ERA5_<Site>/era5_YYYY_MM.nc
        core_outdir = outdir_root / f"ERA5_{site_name}"
        download_era5_monthly_for_area(
            outdir=core_outdir,
            area=area,
            start_date=start_date,
            end_date=end_date,
            variables=core_vars,
            filename_template="era5_{year}_{month}.nc",
        )

        # Roughness folder: data/raw/era5/ERA5_<Site>_fsr_zust/era5_YYYY_MM.nc
        rough_outdir = outdir_root / f"ERA5_{site_name}_fsr_zust"
        download_era5_monthly_for_area(
            outdir=rough_outdir,
            area=area,
            start_date=start_date,
            end_date=end_date,
            variables=rough_vars,
            filename_template="era5_{year}_{month}.nc",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 as per config/sites.yaml (project defaults).")
    parser.add_argument(
        "--config",
        default=str(repo_root() / "config" / "sites.yaml"),
        help="Path to config/sites.yaml",
    )
    args = parser.parse_args()

    run_defaults(Path(args.config))


if __name__ == "__main__":
    main()
