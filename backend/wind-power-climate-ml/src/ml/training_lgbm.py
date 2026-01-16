"""LightGBM training utilities.

This module mirrors the Colab training notebook logic you provided:
  - explicit, fixed feature list
  - Leave-One-Group-Out cross-validation by turbine_id (LOUO)
  - robust median imputation
  - optional end-to-end energy reconstruction evaluation

The functions here are intended to be called from scripts (see scripts/train_lgbm_louo.py)
or imported into notebooks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


def load_ml_data(data_path: str | os.PathLike) -> pd.DataFrame:
    """Load one ML dataset CSV or concatenate all CSVs from a directory."""
    p = Path(data_path)

    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files in {p}")
        dfs: List[pd.DataFrame] = []
        for f in files:
            df = pd.read_csv(f)
            df["__source_file__"] = f.name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    if p.is_file() and p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        df["__source_file__"] = p.name
        return df

    raise ValueError(f"Invalid data_path: {data_path}")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the explicit feature set expected by the pipeline.

    This list matches the ML dataset columns produced by src/ml/dataset_builder.py.
    """

    feature_cols = [
        # Wind
        "era5_ws_10m",
        "era5_ws_100m",
        "era5_ws_hub",
        "era5_wd_10m",
        "era5_wd_100m",
        "era5_shear_alpha",
        # Thermodynamics
        "era5_t2m_K",
        "era5_d2m_K",
        "era5_sp_Pa",
        "era5_air_density_kgm3",
        # Surface / turbulence
        "era5_surface_roughness_m",
        "era5_friction_velocity_ms",
        # Time encodings
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        # Turbine / site
        "hub_height_m",
        "rated_power_kw",
        "rotor_diameter_m",
        "specific_power_wpm2",
        "lat",
        "lon",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return feature_cols


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def train_louo(
    *,
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: str | os.PathLike,
    group_col: str = "turbine_id",
    target_col: str = "target_log_correction_factor",
    valid_col: str = "target_is_valid",
    random_state: int = 42,
    lgbm_params: Dict[str, Any] | None = None,
) -> Tuple[Any, SimpleImputer, pd.DataFrame]:
    """Train LightGBM with LOUO by turbine_id and persist artifacts.

    Returns: (final_model, fitted_imputer, cv_metrics_df)
    """

    # Lazy import so the module can be imported without LightGBM installed.
    import lightgbm as lgb  # type: ignore
    import joblib  # type: ignore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Keep only valid training rows
    train = df[df[valid_col].astype(bool)].copy()
    train = train[np.isfinite(train[target_col].astype(float).values)]
    if train.empty:
        raise ValueError(
            "No training rows after filtering by target_is_valid and finite target. "
            "Check your dataset_builder filters and target selection."
        )

    X = train[feature_cols].copy()
    y = train[target_col].astype(float).values
    groups = train[group_col].astype(str).values

    # Robustness
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    params = dict(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    if lgbm_params:
        params.update(lgbm_params)

    logo = LeaveOneGroupOut()
    rows: List[Dict[str, Any]] = []

    rs = np.random.RandomState(random_state)

    for fold, (tr_idx, te_idx) in enumerate(logo.split(X_imp, y, groups=groups), start=1):
        held_out = groups[te_idx][0]

        X_tr, y_tr = X_imp[tr_idx], y[tr_idx]
        X_te, y_te = X_imp[te_idx], y[te_idx]

        # internal val split for early stopping (from training only)
        n_tr = X_tr.shape[0]
        val_size = min(max(2000, int(0.1 * n_tr)), max(1000, n_tr // 5))
        val_idx = rs.choice(n_tr, size=val_size, replace=False)

        X_val, y_val = X_tr[val_idx], y_tr[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)],
        )

        pred = model.predict(X_te)
        rmse, mae, r2 = _metrics(y_te, pred)

        rows.append(
            {
                "fold": fold,
                "held_out_turbine": held_out,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "rmse_logcf": rmse,
                "mae_logcf": mae,
                "r2_logcf": r2,
                "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
            }
        )

    cv = pd.DataFrame(rows).sort_values("held_out_turbine").reset_index(drop=True)

    # Fit final model on all training data
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X_imp, y)

    # Save artifacts
    joblib.dump(final_model, out_dir / "final_model.pkl")
    joblib.dump(imputer, out_dir / "feature_imputer.pkl")
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    cv.to_csv(out_dir / "cv_metrics.csv", index=False)

    return final_model, imputer, cv


def end_to_end_energy_eval(
    *,
    df: pd.DataFrame,
    model: Any,
    imputer: SimpleImputer,
    feature_cols: List[str],
    valid_col: str = "target_is_valid",
) -> Dict[str, float] | None:
    """Reconstruct energy as expected_energy_kwh * exp(pred_log_cf) and score on valid rows."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce", utc=True)

    needed = ["energy_kwh", "expected_energy_kwh"]
    if not all(c in tmp.columns for c in needed):
        return None

    X_all = imputer.transform(tmp[feature_cols])
    tmp["pred_log_cf"] = model.predict(X_all)

    m = (
        tmp[valid_col].astype(bool)
        & tmp["energy_kwh"].notna()
        & tmp["expected_energy_kwh"].notna()
        & np.isfinite(tmp["pred_log_cf"].values)
    )

    tmp = tmp.loc[m].copy()
    if tmp.empty:
        return None

    tmp["pred_energy_kwh"] = tmp["expected_energy_kwh"].astype(float) * np.exp(tmp["pred_log_cf"].astype(float))

    y_true = tmp["energy_kwh"].astype(float).values
    y_pred = tmp["pred_energy_kwh"].astype(float).values

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"rmse_kwh": rmse, "mae_kwh": mae, "r2": r2}
