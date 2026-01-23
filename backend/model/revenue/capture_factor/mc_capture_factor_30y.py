"""
mc_capture_factor_30y.py

Monte Carlo forecast of wind capture factor (CF) for 30 years (monthly),
based on your regression:

    z_t = log(CF_t)
    z_t = const + phi*z_{t-1} + beta*log(wind_mwh_t) + season(month_t) + eta_t

Where:
- CF_t is positive (we model in log-space)
- wind_mwh_t is a monthly "national wind proxy" (or similar)
- seasonality is via month-of-year dummies
- eta_t are residuals (unexplained effects like market structure, congestion, etc.)

Monte Carlo generation:
- Fit model on history, compute residuals eta_hat_t.
- Generate future eta_t via block bootstrap (preserves persistence/regimes).
- Build future wind proxy via:
    - repeating seasonal pattern from a reference window
    - deterministic growth scenario (low/base/high)
- Recursively forecast z_t with the bootstrapped shocks.

Outputs:
- capture_factor_mc_30y_monthly_paths.csv (long format)
  columns: date, sim, scenario, wind_mwh, capture_factor

Notes:
- We include CF guardrails (floor/cap) AFTER exponentiating, as you already do.
- This script simulates residual uncertainty; it does NOT (yet) sample coefficient uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

# =========================
# CONFIG
# =========================
INPUT_FILE = "capture_factor_monthly_historical.csv"
DATE_COL = "month"
CF_COL = "capture_factor"
WIND_COL = "wind_mwh"

FORECAST_YEARS = 30
N_SIMS = 2000

# Block bootstrap settings for monthly residuals
BLOCK_SIZE_MONTHS = 12
RANDOM_SEED = 123

# Reference window for constructing repeating seasonal wind pattern
REF_START = "2020-01-01"
REF_END = "2024-12-01"

# Wind growth scenarios
WIND_GROWTH_SCEN = {
    "low": 0.005,
    "base": 0.015,
    "high": 0.030,
}

# CF bounds
CF_FLOOR = 0.55
CF_CAP = 1.15

OUT_FILE = "capture_factor_mc_30y_monthly_paths.csv"

# =========================
# Helpers
# =========================
def to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())


def block_bootstrap_1d(residuals: np.ndarray, out_len: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    res = np.asarray(residuals, dtype=float)
    n = res.size
    if n < block_size:
        raise ValueError(f"Not enough residuals ({n}) for block_size={block_size}")
    max_start = n - block_size
    blocks = []
    while sum(len(b) for b in blocks) < out_len:
        s = int(rng.integers(0, max_start + 1))
        blocks.append(res[s : s + block_size])
    return np.concatenate(blocks)[:out_len]


# =========================
# 1) Load history
# =========================
df = pd.read_csv(INPUT_FILE, parse_dates=[DATE_COL])
df[DATE_COL] = to_month_start(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

df[CF_COL] = pd.to_numeric(df[CF_COL], errors="coerce")
df[WIND_COL] = pd.to_numeric(df[WIND_COL], errors="coerce")
df = df.dropna(subset=[CF_COL, WIND_COL])

# CF must be positive for log-space
df = df[df[CF_COL] > 0].copy()

# =========================
# 2) Fit CF regression in log-space
# =========================
df["z"] = np.log(df[CF_COL].astype(float).values)
df["z_lag"] = df["z"].shift(1)
df["month_num"] = df[DATE_COL].dt.month.astype(int)

# Seasonality via month dummies (month 1 baseline)
month_dummies = pd.get_dummies(df["month_num"], prefix="m", drop_first=True).astype(float)

# Wind driver (log) - avoid log(0)
df["log_wind"] = np.log(np.maximum(df[WIND_COL].astype(float).values, 1.0))

X = pd.concat([df["z_lag"], df["log_wind"], month_dummies], axis=1)
X = sm.add_constant(X)

reg = pd.concat([df["z"], X], axis=1).dropna()
model = sm.OLS(reg["z"].astype(float), reg[X.columns].astype(float)).fit()
print(model.summary())

params = model.params
const = float(params["const"])
phi = float(params["z_lag"])
beta_wind = float(params["log_wind"])
season_params = {k: float(v) for k, v in params.items() if k.startswith("m_")}

# Optional stability clamp (keeps AR from drifting)
phi = float(np.clip(phi, 0.0, 0.98))

# Residuals (eta_hat)
z_hat = (
    const
    + phi * reg["z_lag"].values
    + beta_wind * reg["log_wind"].values
    + np.sum([season_params.get(c, 0.0) * reg[c].values for c in month_dummies.columns], axis=0)
)
eta_hat = (reg["z"].values - z_hat).astype(float)

print("\n--- Fitted CF model ---")
print(f"phi={phi:.4f}, beta_wind={beta_wind:.4f}, residuals={len(eta_hat)} block={BLOCK_SIZE_MONTHS}")

# =========================
# 3) Build repeating monthly wind seasonal pattern from reference window
# =========================
mask_ref = (df[DATE_COL] >= pd.Timestamp(REF_START)) & (df[DATE_COL] <= pd.Timestamp(REF_END))
if mask_ref.sum() < 24:
    raise ValueError("Reference window too short/missing. Adjust REF_START/REF_END.")

ref = df.loc[mask_ref, [DATE_COL, WIND_COL]].copy()
ref["m"] = ref[DATE_COL].dt.month.astype(int)

# Seasonal wind profile: average wind_mwh by month-of-year
wind_seasonal = ref.groupby("m")[WIND_COL].mean().to_dict()

# Anchor wind level: mean over reference window
wind_anchor = float(ref[WIND_COL].mean())
anchor_date = ref[DATE_COL].iloc[-1]

print(f"Wind anchor={wind_anchor:,.0f} MWh/month @ {anchor_date.date()}  (ref {REF_START}..{REF_END})")

# =========================
# 4) Future horizon
# =========================
last_date = df[DATE_COL].max()
horizon_months = FORECAST_YEARS * 12
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")

# Start from last observed state
z0 = float(df["z"].iloc[-1])

rng = np.random.default_rng(RANDOM_SEED)

def season_effect(month: int) -> float:
    return float(season_params.get(f"m_{month}", 0.0))  # month 1 baseline 0.0

# =========================
# 5) Monte Carlo per wind-growth scenario
# =========================
rows = []

for scen, g in WIND_GROWTH_SCEN.items():
    g = float(g)

    # Deterministic future wind proxy for this scenario (same for all sims)
    wind_future = np.zeros(horizon_months, dtype=float)
    for i, d in enumerate(future_dates):
        months_from_anchor = (d.year - anchor_date.year) * 12 + (d.month - anchor_date.month)
        growth_factor = (1.0 + g) ** (months_from_anchor / 12.0)

        # Seasonal baseline for this month-of-year
        base = float(wind_seasonal[int(d.month)])

        # Apply growth. (If you want to re-scale to exact mean=wind_anchor, we can add it.)
        wind_future[i] = base * growth_factor

    log_wind_future = np.log(np.maximum(wind_future, 1.0))

    for sim in range(1, N_SIMS + 1):
        # Block-bootstrap residual shocks for CF
        eta_path = block_bootstrap_1d(eta_hat, out_len=horizon_months, block_size=BLOCK_SIZE_MONTHS, rng=rng)

        z_prev = z0
        for i, d in enumerate(future_dates):
            s_t = season_effect(int(d.month))

            # Recursive state update with stochastic residual
            z_t = const + phi * z_prev + beta_wind * float(log_wind_future[i]) + s_t + float(eta_path[i])

            cf = float(np.exp(z_t))
            # Guardrails in price-space (as you already do)
            cf = float(np.clip(cf, CF_FLOOR, CF_CAP))

            rows.append((d, sim, scen, float(wind_future[i]), cf))
            z_prev = z_t

out = pd.DataFrame(rows, columns=["date", "sim", "scenario", "wind_mwh", "capture_factor"])
out.to_csv(OUT_FILE, index=False)
print(f"\n[CF] Wrote {len(out):,} rows -> {OUT_FILE}")
