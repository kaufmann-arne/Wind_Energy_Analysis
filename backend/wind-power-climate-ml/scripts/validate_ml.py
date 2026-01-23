#!/usr/bin/env python
"""CLI wrapper for ML dataset validation."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

import pandas as pd

from src.ml.validation import validate_ml_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--climate-prefix", default="era5_")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    report = validate_ml_dataset(df, climate_prefix=args.climate_prefix)
    print(report)


if __name__ == "__main__":
    main()
