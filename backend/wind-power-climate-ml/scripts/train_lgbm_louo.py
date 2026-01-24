"""Train the correction-factor model (LightGBM) with Leave-One-Turbine-Out CV.

Supports optional Random Search:
- optimize chosen metric over LOUO folds
- always trains target_log_correction_factor
- reports energy metrics (hourly + monthly sums) when energy columns exist
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd

from ml.training_lgbm import (
    end_to_end_energy_eval,
    get_feature_cols,
    load_ml_data,
    random_search_louo,
    save_artifacts,
    train_final_model,
    compute_monthly_logcf_baseline,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True, help="Directory of ML CSVs or a single CSV file.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for model artifacts.")
    p.add_argument("--group-col", type=str, default="turbine_id")
    p.add_argument("--target-col", type=str, default="target_log_correction_factor")
    p.add_argument("--valid-col", type=str, default="target_is_valid")
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--random-search", action="store_true", help="Enable random search hyperparameter optimization.")
    p.add_argument("--n-iter", type=int, default=25, help="Number of random search iterations.")
    p.add_argument(
        "--opt-metric",
        type=str,
        default="rmse_logcf",
        choices=[
            "rmse_logcf",
            "rmse_energy_hourly_kwh",
            "rmse_energy_monthly_kwh",
        ],
        help="Metric minimized during random search .",
    )
    p.add_argument("--skip-energy-eval", action="store_true", help="Skip end-to-end energy evaluation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_all: pd.DataFrame = load_ml_data(args.data_path)
    feature_cols = get_feature_cols(df_all)
    baseline = compute_monthly_logcf_baseline(
        df_all,
        valid_col=args.valid_col,
        target_col=args.target_col,
    )
    # Default baseline params 
    base_params = dict(
        n_estimators=8000,        # early stopping in CV; final will use tuned n_estimators_final
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.random_state,
        n_jobs=-1,
    )

    if args.random_search:
        best_params, best_cv, search_results, best_mean_best_it = random_search_louo(
            df=df_all,
            feature_cols=feature_cols,
            group_col=args.group_col,
            target_col=args.target_col,
            valid_col=args.valid_col,
            random_state=args.random_state,
            n_iter=args.n_iter,
            objective_metric=args.opt_metric,
            objective_agg="worst",
        )

        # Use mean best_iteration from CV to set final n_estimators 
        n_estimators_final = int(max(200, round(best_mean_best_it * 1.05)))
        model, imputer = train_final_model(
            df=df_all,
            feature_cols=feature_cols,
            group_col=args.group_col,
            target_col=args.target_col,
            valid_col=args.valid_col,
            random_state=args.random_state,
            lgbm_params=best_params,
            n_estimators_final=n_estimators_final,
        )

        save_artifacts(
            out_dir=args.out_dir,
            model=model,
            imputer=imputer,
            feature_cols=feature_cols,
            cv_table=best_cv,
            best_params={**best_params, "n_estimators_final": n_estimators_final, "opt_metric": args.opt_metric},
            search_results=search_results,
            baseline=baseline,
        )

        print("\nRandom search completed.")
        print(f"Worst-fold {args.opt_metric}: {best_cv[args.opt_metric].max():.6f}")
        print(f"Final n_estimators: {n_estimators_final}")

    else:
        model, imputer = train_final_model(
            df=df_all,
            feature_cols=feature_cols,
            group_col=args.group_col,
            target_col=args.target_col,
            valid_col=args.valid_col,
            random_state=args.random_state,
            lgbm_params=base_params,
            n_estimators_final=4000,
        )

        from ml.training_lgbm import _run_louo_cv  # type: ignore
        train_df = df_all[df_all[args.valid_col].astype(bool)].copy()
        train_df = train_df[pd.notna(train_df[args.target_col])].copy()
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce", utc=True)

        cv_res = _run_louo_cv(
            train_df=train_df,
            feature_cols=feature_cols,
            group_col=args.group_col,
            target_col=args.target_col,
            random_state=args.random_state,
            lgbm_params=base_params,
            objective_metric="rmse_logcf",
        )

        save_artifacts(
            out_dir=args.out_dir,
            model=model,
            imputer=imputer,
            baseline=baseline,
            feature_cols=feature_cols,
            cv_table=cv_res.cv_table,
            best_params={**base_params, "n_estimators_final": 4000, "opt_metric": "rmse_logcf"},
            search_results=None,
        )

    if not args.skip_energy_eval:
        energy_report = end_to_end_energy_eval(df=df_all, model=model, imputer=imputer, feature_cols=feature_cols)
        if energy_report is not None:
            print("\nEnd-to-end energy metrics (valid rows):")
            for k, v in energy_report.items():
                print(f"  {k}: {v:.6f}")

    out_dir = Path(args.out_dir)
    print("\nArtifacts written to:", out_dir)
    for f in [
        out_dir / "final_model.pkl",
        out_dir / "feature_imputer.pkl",
        out_dir / "feature_cols.json",
        out_dir / "best_params.json",
        out_dir / "cv_metrics.csv",
        out_dir / "random_search_results.csv",
    ]:
        if f.exists():
            print(" -", f)


if __name__ == "__main__":
    main()