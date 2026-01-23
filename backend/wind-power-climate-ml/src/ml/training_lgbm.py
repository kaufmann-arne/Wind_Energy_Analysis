"""LightGBM training utilities with LOUO CV + optional random search.

Key design:
- Train target: target_log_correction_factor (logcf)
- Report metrics: logcf metrics + reconstructed ENERGY metrics (hourly + monthly sums)
- Optional random search over hyperparameters to minimize a chosen objective metric

Assumes:
- df contains at least: timestamp, turbine_id, target_is_valid, target_log_correction_factor
- For energy metrics: energy_kwh, expected_energy_kwh
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


# Keep consistent with dataset_builder filtering bounds
PRED_LOGCF_MIN = -3.9
PRED_LOGCF_MAX =  2.3

def compute_monthly_logcf_baseline(
    df: pd.DataFrame,
    *,
    valid_col: str = "target_is_valid",
    target_col: str = "target_log_correction_factor",
) -> dict:
    """
    Baseline C: median logcf pro Kalendermonat (1..12) + globaler Median.
    Wird als JSON gespeichert und in der Website-Inferenz als Fallback/Blend genutzt.
    """
    tmp = df[df[valid_col].astype(bool)].copy()
    tmp = tmp[np.isfinite(tmp[target_col].astype(float).values)]
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=["timestamp"])

    tmp["month"] = tmp["timestamp"].dt.month

    month_median = (
        tmp.groupby("month")[target_col]
        .median()
        .reindex(range(1, 13))
        .astype(float)
    )

    global_median = float(tmp[target_col].median())


    month_median = month_median.fillna(global_median)

    return {
        "baseline_monthly_logcf": month_median.tolist(), 
        "baseline_global_logcf": global_median,
        "baseline_stat": "median",
    }

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

    NOTE: lat/lon intentionally excluded (your decision).
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
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return feature_cols


def _rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def _rmse_mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _monthly_sum_metrics(
    ts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_hours_per_month: int = 24 * 28,
) -> Tuple[float, float, float]:
    """
    Aggregate to monthly sums and compute metrics.
    Only months with >= min_hours_per_month samples are used (production-consistent).
    Uses Period(M) grouping (avoids timezone-drop warnings).
    """
    tmp = pd.DataFrame({"timestamp": ts.values, "y_true": y_true, "y_pred": y_pred})
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=["timestamp"])

    # Group by calendar month as Period to avoid tz warnings
    tmp["month"] = tmp["timestamp"].dt.tz_convert(None).dt.to_period("M")

    g = (
        tmp.groupby("month", as_index=False)
        .agg(
            y_true_sum=("y_true", "sum"),
            y_pred_sum=("y_pred", "sum"),
            n=("y_true", "size"),
        )
    )

    # Production-like coverage filter
    g = g[g["n"] >= int(min_hours_per_month)].copy()

    if len(g) == 0:
        return float("nan"), float("nan"), float("nan")

    if len(g) < 2:
        rmse = float(np.sqrt(mean_squared_error(g["y_true_sum"], g["y_pred_sum"])))
        mae = float(mean_absolute_error(g["y_true_sum"], g["y_pred_sum"]))
        return rmse, mae, float("nan")

    return _rmse_mae_r2(g["y_true_sum"].values, g["y_pred_sum"].values)


def _reconstruct_energy(
    expected_energy_kwh: np.ndarray,
    pred_logcf: np.ndarray,
    *,
    rated_power_kw: Optional[np.ndarray] = None,
    cap_factor: float = 1.05,
) -> np.ndarray:
    """
    pred_energy = expected_energy_kwh * exp(clipped_logcf)
    Optional production-consistent cap: <= cap_factor * rated_power_kw
    """
    pred_logcf = np.clip(pred_logcf, PRED_LOGCF_MIN, PRED_LOGCF_MAX)
    pred_energy = expected_energy_kwh.astype(float) * np.exp(pred_logcf.astype(float))

    if rated_power_kw is not None:
        cap = cap_factor * rated_power_kw.astype(float)
        pred_energy = np.minimum(pred_energy, cap)

    return pred_energy



@dataclass
class CVResult:
    cv_table: pd.DataFrame
    mean_objective: float
    worst_objective: float
    p90_objective: float
    mean_best_iteration: float


def _apply_lgb_defaults(params: Dict[str, Any], *, random_state: int) -> Dict[str, Any]:
    """
    Fill-only defaults: setzt nur Keys, die NICHT in params vorhanden sind.
    Überschreibt also keine Random-Search-Parameter.
    """
    out = dict(params)

    defaults = {
        "n_estimators": 8000,     # CV nutzt early stopping; final setzt ggf. n_estimators_final
        "learning_rate": 0.03,
        "num_leaves": 64,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "min_split_gain": 0.0,
        "n_jobs": -1,
        "verbosity": -1,          # unterdrückt "No further splits..." Spam
        "random_state": random_state,
    }

    for k, v in defaults.items():
        out.setdefault(k, v)

    # sklearn-wrapper akzeptiert random_state; in LightGBM heißt es intern seed
    out["random_state"] = random_state
    return out


def _run_louo_cv(
    *,
    train_df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    target_col: str,
    random_state: int,
    lgbm_params: Dict[str, Any],
    objective_metric: str,
) -> CVResult:
    """One LOUO CV run for a given param set. Returns fold table + mean objective."""
    import lightgbm as lgb  # type: ignore

    lgbm_params = _apply_lgb_defaults(lgbm_params, random_state=random_state)

    # X/y/groups
    X = train_df[feature_cols].copy()
    y = train_df[target_col].astype(float).values
    groups = train_df[group_col].astype(str).values
    # Timestamp for blocked val split (preferred). If missing, we fallback to random.
    ts_all_for_split = None
    if "timestamp" in train_df.columns:
        ts_all_for_split = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
    # --- Energy arrays (für Reporting) ---
    has_energy = ("energy_kwh" in train_df.columns) and ("expected_energy_kwh" in train_df.columns) and ("timestamp" in train_df.columns)
    energy_true_all = train_df["energy_kwh"].astype(float).values if has_energy else None
    energy_exp_all = train_df["expected_energy_kwh"].astype(float).values if has_energy else None
    ts_all = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce") if has_energy else None

    logo = LeaveOneGroupOut()
    rows: List[Dict[str, Any]] = []
    rng = np.random.RandomState(random_state)
    best_iters: List[int] = []

    for fold, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        held_out = groups[te_idx][0]

        # Fold-spezifischer Imputer (wichtig!)
        imputer_fold = SimpleImputer(strategy="median")
        X_tr_raw = X.iloc[tr_idx]
        X_te_raw = X.iloc[te_idx]
        X_tr = imputer_fold.fit_transform(X_tr_raw)
        X_te = imputer_fold.transform(X_te_raw)

        y_tr, y_te = y[tr_idx], y[te_idx]

        # internal val split innerhalb Trainingsfold
        n_tr = X_tr.shape[0]
        val_size = min(max(2000, int(0.1 * n_tr)), max(1000, n_tr // 5))
        if val_size >= n_tr:
            # safety: falls fold extrem klein ist
            val_size = max(1, n_tr // 5)

        # --- blocked-by-time validation split (preferred) ---
        if ts_all_for_split is not None:
            tr_ts = ts_all_for_split.iloc[tr_idx].to_numpy(dtype="datetime64[ns]")
            # treat NaT as very old so it doesn't end up in the "latest" validation block
            nat_mask = pd.isna(tr_ts)
            if np.any(nat_mask):
                tr_ts = tr_ts.copy()
                tr_ts[nat_mask] = np.datetime64("1900-01-01")


            order = np.argsort(tr_ts)  
            val_pos = order[-val_size:]
            tr_pos = order[:-val_size]
        else:
            # fallback: random split
            val_pos = rng.choice(n_tr, size=val_size, replace=False)
            tr_mask = np.ones(n_tr, dtype=bool)
            tr_mask[val_pos] = False
            tr_pos = np.where(tr_mask)[0]

        X_tr2, y_tr2 = X_tr[tr_pos], y_tr[tr_pos]
        X_val, y_val = X_tr[val_pos], y_tr[val_pos]

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(
            X_tr2, y_tr2,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)],
        )

        pred_logcf = model.predict(pd.DataFrame(X_te, columns=feature_cols))
        rmse_logcf, mae_logcf = _rmse_mae(y_te, pred_logcf)

        best_it = int(getattr(model, "best_iteration_", lgbm_params.get("n_estimators", 0)))
        best_iters.append(best_it)

        row: Dict[str, Any] = {
            "fold": fold,
            "held_out_turbine": held_out,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "rmse_logcf": rmse_logcf,
            "mae_logcf": mae_logcf,
            "best_iteration": best_it,
        }

        # --- Energy metrics (hourly + monthly sums) ---
        if has_energy and energy_true_all is not None and energy_exp_all is not None and ts_all is not None:
            e_true = energy_true_all[te_idx]
            e_exp = energy_exp_all[te_idx]
            ts_te = ts_all.iloc[te_idx]

            rated_te = None
            if "rated_power_kw" in train_df.columns:
                rated_te = train_df["rated_power_kw"].astype(float).values[te_idx]

            pred_energy = _reconstruct_energy(
                e_exp,
                pred_logcf,
                rated_power_kw=rated_te,
                cap_factor=1.05,
            )

            # ---------- (2) ROBUST FILTER: drop NaN/inf BEFORE metrics ----------
            m = (
                np.isfinite(e_true)
                & np.isfinite(e_exp)
                & np.isfinite(pred_energy)
                & ts_te.notna().to_numpy()
            )

            e_true2 = e_true[m]
            pred_energy2 = pred_energy[m]
            ts_te2 = ts_te.iloc[np.where(m)[0]]

            if e_true2.size >= 2:
                rmse_e_h, mae_e_h, r2_e_h = _rmse_mae_r2(e_true2, pred_energy2)
                rmse_e_m, mae_e_m, r2_e_m = _monthly_sum_metrics(
                    ts_te2,
                    e_true2,
                    pred_energy2,
                    min_hours_per_month=24 * 28,
                )
            else:
                rmse_e_h = mae_e_h = r2_e_h = float("nan")
                rmse_e_m = mae_e_m = r2_e_m = float("nan")

            row.update(
                {
                    "rmse_energy_hourly_kwh": rmse_e_h,
                    "mae_energy_hourly_kwh": mae_e_h,
                    "r2_energy_hourly": r2_e_h,
                    "rmse_energy_monthly_kwh": rmse_e_m,
                    "mae_energy_monthly_kwh": mae_e_m,
                    "r2_energy_monthly": r2_e_m,
                }
            )

        rows.append(row)

    cv = pd.DataFrame(rows).sort_values("held_out_turbine").reset_index(drop=True)

    if objective_metric not in cv.columns:
        raise ValueError(
            f"objective_metric='{objective_metric}' not present in CV table columns: {list(cv.columns)}"
        )

    mean_obj = float(cv[objective_metric].mean())
    worst_obj = float(cv[objective_metric].max())
    p90_obj = float(cv[objective_metric].quantile(0.90))
    mean_best_it = float(np.mean(best_iters)) if best_iters else float(lgbm_params.get("n_estimators", 0))

    return CVResult(
        cv_table=cv,
        mean_objective=mean_obj,
        worst_objective=worst_obj,
        p90_objective=p90_obj,
        mean_best_iteration=mean_best_it,
    )


def _sample_params(rng: np.random.RandomState) -> Dict[str, Any]:
    """Reasonable random-search space for small turbine LOUO setup."""
    # log-uniform helper
    def logu(lo: float, hi: float) -> float:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    # choose depth sometimes unlimited
    max_depth = int(rng.choice([-1, 4, 5, 6, 7, 8, 10, 12]))

    params = dict(
        # keep large; early stopping will find best_iteration
        n_estimators=8000,
        learning_rate=logu(0.01, 0.07),
        num_leaves=int(rng.randint(31, 256)),
        max_depth=max_depth,
        min_child_samples=int(rng.randint(10, 200)),
        subsample=float(rng.uniform(0.6, 1.0)),
        subsample_freq=1,
        colsample_bytree=float(rng.uniform(0.6, 1.0)),
        reg_alpha=logu(1e-4, 1.0),
        reg_lambda=logu(1e-2, 10.0),
        min_split_gain=float(rng.uniform(0.0, 0.2)),
        n_jobs=-1,
    )
    return params


def random_search_louo(
    *,
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    target_col: str,
    valid_col: str,
    random_state: int,
    n_iter: int,
    objective_metric: str,
    objective_agg: str = "mean",   # "mean" | "worst" | "p90"
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, float]:
    """Random search over hyperparameters using LOUO CV.

    Returns:
      best_params, best_cv_table, search_results_table, best_mean_best_iteration
    """
    # Keep only valid training rows
    train = df[df[valid_col].astype(bool)].copy()
    train = train[np.isfinite(train[target_col].astype(float).values)]
    if train.empty:
        raise ValueError("No training rows after filtering. Check target_is_valid and target_col.")

    # Require timestamp for monthly metric
    train["timestamp"] = pd.to_datetime(train["timestamp"], errors="coerce", utc=True)

    rng = np.random.RandomState(random_state)

    best_params: Optional[Dict[str, Any]] = None
    best_cv: Optional[pd.DataFrame] = None
    best_score = float("inf")
    best_mean_best_it = 0.0

    search_rows: List[Dict[str, Any]] = []

    for i in range(1, n_iter + 1):
        params = _sample_params(rng)

        res = _run_louo_cv(
            train_df=train,
            feature_cols=feature_cols,
            group_col=group_col,
            target_col=target_col,
            random_state=random_state,
            lgbm_params=params,
            objective_metric=objective_metric,
        )
        if objective_agg == "mean":
            score = res.mean_objective
        elif objective_agg == "worst":
            score = res.worst_objective
        elif objective_agg == "p90":
            score = res.p90_objective
        else:
            raise ValueError("objective_agg must be one of: 'mean', 'worst', 'p90'")
        row = {
            "iter": i,
            "objective_agg": objective_agg,
            "mean_objective": res.mean_objective,
            "worst_objective": res.worst_objective,
            "p90_objective": res.p90_objective,
            "score_used": score,
        }
        # flatten a subset of params for the table
        for k in [
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_split_gain",
            "n_estimators",
        ]:
            row[k] = params.get(k)
        search_rows.append(row)

        if score < best_score:
            best_score = score
            best_params = params
            best_cv = res.cv_table
            best_mean_best_it = res.mean_best_iteration

    if best_params is None or best_cv is None:
        raise RuntimeError("Random search failed to produce a best model.")

    search_results = pd.DataFrame(search_rows).sort_values("score_used").reset_index(drop=True)
    return best_params, best_cv, search_results, best_mean_best_it


def train_final_model(
    *,
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    target_col: str,
    valid_col: str,
    random_state: int,
    lgbm_params: Dict[str, Any],
    n_estimators_final: Optional[int] = None,
) -> Tuple[Any, SimpleImputer]:
    """Train final model on all valid rows (no early stopping)."""
    import lightgbm as lgb  # type: ignore

    train = df[df[valid_col].astype(bool)].copy()
    train = train[np.isfinite(train[target_col].astype(float).values)]
    if train.empty:
        raise ValueError("No training rows after filtering. Check target_is_valid and target_col.")

    X = train[feature_cols].copy()
    y = train[target_col].astype(float).values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    params_fit = _apply_lgb_defaults(lgbm_params, random_state=random_state)
    if n_estimators_final is not None:
        params_fit["n_estimators"] = int(n_estimators_final)

    model = lgb.LGBMRegressor(**params_fit)
    model.fit(X_imp, y)

    return model, imputer


def save_artifacts(
    *,
    out_dir: str | os.PathLike,
    model: Any,
    imputer: SimpleImputer,
    feature_cols: List[str],
    cv_table: pd.DataFrame,
    best_params: Dict[str, Any],
    search_results: Optional[pd.DataFrame] = None,
    baseline: dict | None = None,   
) -> None:
    import joblib  # type: ignore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "final_model.pkl")
    joblib.dump(imputer, out_dir / "feature_imputer.pkl")
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, default=float))

    cv_table.to_csv(out_dir / "cv_metrics.csv", index=False)
    if search_results is not None:
        search_results.to_csv(out_dir / "random_search_results.csv", index=False)

    if baseline is not None:
        (out_dir / "baseline_monthly_logcf.json").write_text(json.dumps(baseline, indent=2))


def end_to_end_energy_eval(
    *,
    df: pd.DataFrame,
    model: Any,
    imputer: SimpleImputer,
    feature_cols: List[str],
    valid_col: str = "target_is_valid",
) -> Dict[str, float] | None:
    """Evaluate reconstructed energy on all valid rows (hourly + monthly sums)."""
    needed = ["timestamp", "energy_kwh", "expected_energy_kwh"]
    if not all(c in df.columns for c in needed):
        return None

    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce", utc=True)

    m = (
        tmp[valid_col].astype(bool)
        & tmp["energy_kwh"].notna()
        & tmp["expected_energy_kwh"].notna()
    )
    tmp = tmp.loc[m].copy()
    if tmp.empty:
        return None

    X_all = imputer.transform(tmp[feature_cols])
    pred_logcf = model.predict(pd.DataFrame(X_all, columns=feature_cols, index=tmp.index))

    rated = tmp["rated_power_kw"].astype(float).values if "rated_power_kw" in tmp.columns else None
    pred_energy = _reconstruct_energy(
        tmp["expected_energy_kwh"].astype(float).values,
        pred_logcf,
        rated_power_kw=rated,
        cap_factor=1.05,
    )
    y_true = tmp["energy_kwh"].astype(float).values
    m2 = (
        np.isfinite(y_true)
        & np.isfinite(pred_energy)
        & tmp["timestamp"].notna().to_numpy()
    )
    
    y_true2 = y_true[m2]
    pred_energy2 = pred_energy[m2]
    ts2 = tmp["timestamp"].iloc[np.where(m2)[0]]

    if y_true2.size < 2:
        return {
            "rmse_energy_hourly_kwh": float("nan"),
            "mae_energy_hourly_kwh": float("nan"),
            "r2_energy_hourly": float("nan"),
            "rmse_energy_monthly_kwh": float("nan"),
            "mae_energy_monthly_kwh": float("nan"),
            "r2_energy_monthly": float("nan"),
        }

    rmse_h, mae_h, r2_h = _rmse_mae_r2(y_true2, pred_energy2)
    rmse_m, mae_m, r2_m = _monthly_sum_metrics(ts2, y_true2, pred_energy2, min_hours_per_month=24 * 28)

    return {
        "rmse_energy_hourly_kwh": rmse_h,
        "mae_energy_hourly_kwh": mae_h,
        "r2_energy_hourly": r2_h,
        "rmse_energy_monthly_kwh": rmse_m,
        "mae_energy_monthly_kwh": mae_m,
        "r2_energy_monthly": r2_m,
    }