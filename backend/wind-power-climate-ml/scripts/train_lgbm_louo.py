"""Train the correction-factor model (LightGBM) with Leave-One-Turbine-Out CV.

This script is the repo-friendly equivalent of the Colab notebook you provided.

Example:
  python scripts/train_lgbm_louo.py \
    --data-path Data/ML \
    --out-dir  artifacts/LGBM_Model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ml.training_lgbm import (
    end_to_end_energy_eval,
    get_feature_cols,
    load_ml_data,
    train_louo,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Directory containing ML CSVs, or a single CSV file.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for model artifacts.",
    )
    p.add_argument("--group-col", type=str, default="turbine_id")
    p.add_argument("--target-col", type=str, default="target_log_correction_factor")
    p.add_argument("--valid-col", type=str, default="target_is_valid")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--skip-energy-eval",
        action="store_true",
        help="Skip end-to-end energy evaluation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_all: pd.DataFrame = load_ml_data(args.data_path)
    feature_cols = get_feature_cols(df_all)

    model, imputer, cv = train_louo(
        df_all,
        feature_cols=feature_cols,
        out_dir=args.out_dir,
        group_col=args.group_col,
        target_col=args.target_col,
        valid_col=args.valid_col,
        random_state=args.random_state,
    )

    if not args.skip_energy_eval:
        _ = end_to_end_energy_eval(df_all, model, imputer, feature_cols)

    # small convenience printout
    out_dir = Path(args.out_dir)
    print("\nArtifacts:")
    for f in [
        out_dir / "final_model.pkl",
        out_dir / "feature_imputer.pkl",
        out_dir / "feature_cols.json",
        out_dir / "cv_metrics.csv",
    ]:
        print(" -", f)


if __name__ == "__main__":
    main()
