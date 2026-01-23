"""
mc_curtailment_rate_30y.py

Monte Carlo forecast of quarterly curtailment rate (CR) by TSO for 30 years.

This version expects a *proxy-generated* quarterly curtailment series as input:
    curtailment_rate_quarterly_proxy_by_tso.csv

That proxy series is treated as the historical curtailment baseline (2015..2025).
Observed "real" curtailment is NOT required by this script anymore; it was used upstream
only to validate/calibrate the proxy model.

Model on history
----------------
- Build quarterly wind (daily -> quarterly sum)
- Build quarterly load (monthly -> quarterly sum)
- Read quarterly proxy curtailment (already quarterly, wide)
- Define congestion proxy:
      S = wind_mwh / load_mwh
- Fit:
      cr ~ S * C(TSO)

Forecast structure
------------------
- Create deterministic future wind and load:
      level  = mean(last LEVEL_QTRS quarters) per TSO
      season = quarter-of-year factors per TSO (normalized to mean=1)
      growth = (1+g)^(years_ahead)
- Compute future S and predict baseline:
      cr_hat = model(S, TSO)
- Apply grid-improvement factor (scenario assumption):
      cr_hat *= (1 - improvement)^(years_ahead)

Uncertainty
-----------
- Compute historical residuals (proxy history minus fitted baseline)
- Block-bootstrap residuals *within each TSO* to preserve regime persistence
- Simulate:
      cr_sim = clip(cr_hat + eps_bootstrapped)

Output
------
curtailment_mc_30y_quarterly_paths.csv (long format):
  quarter, sim, TSO, cr, wind_mwh, load_mwh, S, cr_hat
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =============================
# CONFIG
# =============================
WIND_DAILY_FILE = "wind_data/wind_onshore_daily_by_tso.csv"
LOAD_MONTHLY_FILE = "grid_data/tso_grid_load_monthly_wide.csv"

# NEW (proxy-only quarterly curtailment series)
CURTAIL_FILE = "curtailment_rate_quarterly_proxy_by_tso.csv"

FORECAST_YEARS = 30
N_SIMS = 2000

# Block bootstrap settings for quarterly residuals
BLOCK_SIZE_QUARTERS = 8  # ~2 years (regime persistence)
RANDOM_SEED = 999

CR_FLOOR = 0.0
CR_CAP = 0.30

TSOS = ["TenneT", "50Hertz", "Amprion", "TransnetBW"]

LOAD_COL_MAP = {
    "TenneT": "load_mwh_tennet",
    "50Hertz": "load_mwh_50hertz",
    "Amprion": "load_mwh_amprion",
    "TransnetBW": "load_mwh_transnetbw",
}

# Growth assumptions (scenario knobs)
G_WIND = {tso: 0.01 for tso in TSOS}   # +1%/yr wind proxy
G_LOAD = {tso: 0.005 for tso in TSOS}  # +0.5%/yr load proxy

# Grid infrastructure improvements (scenario knobs)
GRID_IMPROVEMENT = {
    "TenneT": 0.02,
    "50Hertz": 0.015,
    "Amprion": 0.01,
    "TransnetBW": 0.01,
}

LEVEL_QTRS = 8  # baseline level from last N quarters
OUT_FILE = "curtailment_mc_30y_quarterly_paths.csv"


# =============================
# HELPERS
# =============================
def to_quarter_period_from_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("Q")


def wide_to_quarter_long(df: pd.DataFrame, date_col: str, value_cols: list[str], value_name: str) -> pd.DataFrame:
    """
    Convert wide daily/monthly -> long quarterly totals.
    - parse date
    - convert to Period('Q')
    - melt into (quarter, TSO, value)
    - sum within quarter per TSO
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["quarter"] = to_quarter_period_from_dates(df[date_col])

    long = df.melt(id_vars="quarter", value_vars=value_cols, var_name="TSO", value_name=value_name)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.dropna(subset=[value_name])

    return long.groupby(["quarter", "TSO"], as_index=False)[value_name].sum()


def make_season_factors(df_q: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Quarter-of-year seasonality per TSO, normalized to mean=1 within each TSO.
    """
    tmp = df_q.copy()
    tmp["q_num"] = tmp["quarter"].dt.quarter.astype(int)

    seas = tmp.groupby(["TSO", "q_num"])[value_col].mean().reset_index()
    seas["mean_tso"] = seas.groupby("TSO")[value_col].transform("mean")
    seas["season_factor"] = seas[value_col] / seas["mean_tso"]

    return seas[["TSO", "q_num", "season_factor"]]


def make_future_series(
    df_q: pd.DataFrame,
    value_col: str,
    g_map: dict[str, float],
    future_q: pd.PeriodIndex,
    last_q: pd.Period,
) -> pd.DataFrame:
    """
    Deterministic future series per TSO:
      value = level(TSO) * season_factor(TSO, q_num) * (1+g)^(years_ahead)

    level(TSO) uses the mean of the last LEVEL_QTRS quarters.
    """
    df_q = df_q.copy().sort_values(["TSO", "quarter"])

    # Baseline level per TSO
    levels = (
        df_q.groupby("TSO", as_index=True)[value_col]
        .apply(lambda s: float(s.tail(LEVEL_QTRS).mean()))
        .to_dict()
    )

    seas = make_season_factors(df_q, value_col)

    future = pd.MultiIndex.from_product([future_q, TSOS], names=["quarter", "TSO"]).to_frame(index=False)
    future["q_num"] = future["quarter"].dt.quarter.astype(int)

    quarters_ahead = future["quarter"].apply(lambda p: p.ordinal) - last_q.ordinal
    years_ahead = quarters_ahead / 4.0

    future = future.merge(seas, on=["TSO", "q_num"], how="left")
    future["season_factor"] = future["season_factor"].fillna(1.0)

    out_parts = []
    for tso in TSOS:
        g = float(g_map[tso])
        lvl = float(levels[tso])

        idx = future["TSO"] == tso
        growth = (1.0 + g) ** years_ahead[idx].values
        vals = lvl * future.loc[idx, "season_factor"].values * growth

        out_parts.append(pd.DataFrame({"quarter": future.loc[idx, "quarter"].values, "TSO": tso, value_col: vals}))

    return pd.concat(out_parts, ignore_index=True).sort_values(["quarter", "TSO"]).reset_index(drop=True)


def block_bootstrap_by_tso(residual_df: pd.DataFrame, tso: str, out_len: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Block bootstrap residuals for one TSO to preserve multi-quarter regimes.
    """
    r = residual_df.loc[residual_df["TSO"] == tso].sort_values("quarter")["resid"].values.astype(float)
    n = r.size

    if n < block_size:
        raise ValueError(f"TSO={tso}: not enough residuals ({n}) for block_size={block_size}")

    max_start = n - block_size
    blocks = []
    while sum(len(b) for b in blocks) < out_len:
        s = int(rng.integers(0, max_start + 1))
        blocks.append(r[s : s + block_size])

    return np.concatenate(blocks)[:out_len]


# =============================
# LOAD & PREPROCESS HISTORY
# =============================

# Wind: daily -> quarterly
wind = pd.read_csv(WIND_DAILY_FILE)
wind_q = wide_to_quarter_long(wind, "date", TSOS, "wind_mwh")

# Load: monthly wide -> rename -> quarterly
load = pd.read_csv(LOAD_MONTHLY_FILE)
load = load.rename(columns={v: k for k, v in LOAD_COL_MAP.items()})
load_q = wide_to_quarter_long(load, "date", TSOS, "load_mwh")

# Curtailment: already quarterly wide (proxy output)
curt = pd.read_csv(CURTAIL_FILE)
curt["quarter"] = pd.PeriodIndex(curt["quarter"].astype(str), freq="Q")

records = []
for tso in TSOS:
    col = f"{tso}_curtailment_rate"
    if col not in curt.columns:
        raise ValueError(f"Missing column '{col}' in {CURTAIL_FILE}")

    records.append(
        pd.DataFrame(
            {
                "quarter": curt["quarter"],
                "TSO": tso,
                "cr": pd.to_numeric(curt[col], errors="coerce"),
            }
        )
    )

curt_q = pd.concat(records, ignore_index=True).dropna(subset=["quarter", "TSO", "cr"])
curt_q["cr"] = curt_q["cr"].clip(CR_FLOOR, CR_CAP)

# Merge modeling frame (quarters where all inputs exist)
hist = (
    curt_q.merge(wind_q, on=["quarter", "TSO"], how="inner")
          .merge(load_q, on=["quarter", "TSO"], how="inner")
)

hist["S"] = hist["wind_mwh"] / hist["load_mwh"]
hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["S"])

if hist.empty:
    raise RuntimeError("Historical modeling frame is empty after merges. Check quarter coverage and file paths.")


# =============================
# FIT MODEL ON HISTORY (proxy baseline)
# =============================
model0 = smf.ols("cr ~ S * C(TSO)", data=hist).fit()
print(model0.summary())

# Historical residuals (used only for uncertainty shaping)
hist["cr_hat"] = model0.predict(hist).astype(float)
hist["resid"] = (hist["cr"].astype(float) - hist["cr_hat"]).astype(float)


# =============================
# FUTURE QUARTERS (30 years)
# =============================
last_q_wind = wind_q["quarter"].max()
last_q_load = load_q["quarter"].max()
last_q_cr = curt_q["quarter"].max()
last_q = max(last_q_wind, last_q_load, last_q_cr)

future_start = last_q + 1
future_end = future_start + (FORECAST_YEARS * 4) - 1
future_q = pd.period_range(future_start, future_end, freq="Q")

# Deterministic wind/load futures (scenario)
wind_f = make_future_series(wind_q, "wind_mwh", G_WIND, future_q=future_q, last_q=last_q)
load_f = make_future_series(load_q, "load_mwh", G_LOAD, future_q=future_q, last_q=last_q)

future = wind_f.merge(load_f, on=["quarter", "TSO"], how="inner")
future["S"] = future["wind_mwh"] / future["load_mwh"]
future = future.replace([np.inf, -np.inf], np.nan).dropna(subset=["S"])

future["cr_hat"] = model0.predict(future).astype(float)

# Apply grid improvement: cr_hat *= (1 - g_imp)^(years_ahead)
quarters_ahead = future["quarter"].apply(lambda p: p.ordinal) - last_q.ordinal
years_ahead = quarters_ahead / 4.0

g_imp_series = future["TSO"].map(GRID_IMPROVEMENT).fillna(0.0).astype(float)
improve_factor = (1.0 - g_imp_series.values) ** years_ahead.values

future["cr_hat"] = (future["cr_hat"].values * improve_factor).clip(CR_FLOOR, CR_CAP)


# =============================
# MONTE CARLO: add bootstrapped residuals per TSO
# =============================
rng = np.random.default_rng(RANDOM_SEED)
horizon_q = len(future_q)

rows = []
for sim in range(1, N_SIMS + 1):
    eps_paths = {
        tso: block_bootstrap_by_tso(hist, tso, out_len=horizon_q, block_size=BLOCK_SIZE_QUARTERS, rng=rng)
        for tso in TSOS
    }

    for tso in TSOS:
        f_tso = future.loc[future["TSO"] == tso].sort_values("quarter").reset_index(drop=True)
        eps = eps_paths[tso]

        cr_sim = (f_tso["cr_hat"].values.astype(float) + eps).clip(CR_FLOOR, CR_CAP)

        for i in range(horizon_q):
            rows.append(
                (
                    str(f_tso.loc[i, "quarter"]),
                    sim,
                    tso,
                    float(cr_sim[i]),
                    float(f_tso.loc[i, "wind_mwh"]),
                    float(f_tso.loc[i, "load_mwh"]),
                    float(f_tso.loc[i, "S"]),
                    float(f_tso.loc[i, "cr_hat"]),
                )
            )

out = pd.DataFrame(rows, columns=["quarter", "sim", "TSO", "cr", "wind_mwh", "load_mwh", "S", "cr_hat"])
out.to_csv(OUT_FILE, index=False)
print(f"\n[CR] Wrote {len(out):,} rows -> {OUT_FILE}")
