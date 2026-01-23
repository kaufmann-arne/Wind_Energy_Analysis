"""
mc_market_price_30y.py

Monte Carlo forecast for German day-ahead *monthly average* prices over a 30-year horizon.

What this script does
---------------------
1) Reads hourly prices from INPUT_FILE and aggregates them to month-start timestamps ("MS").
2) Transforms prices into log-space with a SHIFT so negative prices are still supported:
       y_t = log(P_t + SHIFT)
3) Builds a deterministic long-run mean path (in log space) anchored to a historical window:
       mu_t = log(P_anchor + SHIFT) + (years_from_anchor) * log(1 + annual_drift)
4) Models deviations around that mean (z_t = y_t - mu_t) using:
       z_t = phi*z_{t-1} + seasonality(month) + eps_t
   (We intentionally do NOT propagate the fitted intercept during simulation; see notes below.)
5) Generates future shocks eps_t via a rolling block bootstrap to preserve
   “lumpy” regimes / persistence and heavy tails better than iid noise.
6) Simulates N_SIMS monthly paths and writes them to OUT_FILE.

Output
------
- OUT_FILE is always monthly ("MS").
- By default: columns = [date, sim, price_eur_mwh]
- If WRITE_DEBUG_COLUMNS=True: includes y/mu/z in log space for diagnostics.

Design choices / assumptions
----------------------------
- PRICE_SHIFT must be large enough so (P + SHIFT) > 0 for all historical months.
- ANNUAL_DRIFT is applied to the *mean level* (log space) as a simple deterministic drift.
- Residual stabilization is optional but useful in practice to avoid unrealistic extremes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

# =============================================================================
# CONFIG
# =============================================================================

# Input
INPUT_FILE = "prices_hourly_merged.csv"
DATE_COL = None  # if None -> autodetect
PRICE_COL = "price_eur_mwh"

# Simulation horizon
FORECAST_YEARS = 30
N_SIMS = 2000

# Output
OUT_FILE = "market_price_mc_30y_monthly_paths.csv"
WRITE_DEBUG_COLUMNS = False  # include extra columns for debugging/plots

# Log transform shift:
# Must ensure (price + PRICE_SHIFT) stays strictly positive for ALL historical months.
PRICE_SHIFT = 200.0

# Deterministic drift applied to the anchored mean level (nominal)
ANNUAL_DRIFT = 0.01

# Anchor window (month-start timestamps)
ANCHOR_START = "2018-01-01"
ANCHOR_END = "2020-12-01"

# --- Monte Carlo stability knobs ---
# Block bootstrap: smaller blocks reduce the chance of extremely long regimes repeating.
BLOCK_SIZE_MONTHS = 6

# Residual stabilization:
# - Clip to reduce extreme outliers (winsorization)
# - Scale to tone down volatility if needed
RESID_CLIP_SIGMA = 2.5
RESID_SCALE = 0.70

# Clamp initial deviation (so the simulation doesn’t start from a rare extreme month)
Z0_CLIP_SIGMA = 2.0

# Optional: increase mean reversion slightly during simulation (set 1.0 to disable)
PHI_SIM_SHRINK = 0.90

RANDOM_SEED = 42


# =============================================================================
# Helpers
# =============================================================================
def autodetect_datetime_col(df: pd.DataFrame) -> str:
    """
    Pick a likely datetime column.
    1) Try a small list of common column names.
    2) Fallback: test columns and choose one that parses well.
    """
    candidates = ["date", "datetime", "timestamp", "time", "utc_timestamp", "Datum", "Zeit"]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
            return c

    raise ValueError(f"Could not autodetect datetime column. Columns: {list(df.columns)}")


def to_month_start(s: pd.Series) -> pd.Series:
    """
    Convert timestamps to month-start (MS) timestamps.
    This is used to align the monthly aggregation and the forecast index.
    """
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())


def block_bootstrap_1d(
    residuals: np.ndarray,
    out_len: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Rolling block bootstrap:
    - Randomly sample consecutive blocks of length `block_size`
    - Concatenate until length `out_len` is reached

    Motivation:
    - Keeps short-run persistence in shocks (regime-ish behavior)
    - Keeps fat tails from the empirical residual distribution
    """
    res = np.asarray(residuals, dtype=float)
    n = res.size
    if n < block_size:
        raise ValueError(f"Not enough residuals ({n}) for block_size={block_size}")

    max_start = n - block_size
    blocks = []
    need = out_len

    while need > 0:
        start = int(rng.integers(0, max_start + 1))
        block = res[start : start + block_size]
        blocks.append(block)
        need -= block.size

    return np.concatenate(blocks)[:out_len]


# =============================================================================
# 1) Load hourly data and aggregate to monthly averages
# =============================================================================
df = pd.read_csv(INPUT_FILE)

if DATE_COL is None or DATE_COL not in df.columns:
    DATE_COL = autodetect_datetime_col(df)
    print(f"[PRICE] Using datetime column: {DATE_COL}")

# Parse + basic cleaning
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
df = df.dropna(subset=[PRICE_COL])

# Month-start aggregation ("MS"):
# - keep "date" as an actual timestamp for easy joins/plots later
df["date"] = to_month_start(df[DATE_COL])
monthly = (
    df.groupby("date", as_index=False)[PRICE_COL]
    .mean()
    .rename(columns={PRICE_COL: "price_eur_mwh"})
    .sort_values("date")
    .reset_index(drop=True)
)

# Log transform prep
monthly["price_shifted"] = monthly["price_eur_mwh"] + float(PRICE_SHIFT)
if (monthly["price_shifted"] <= 0).any():
    # If this happens, the shift is invalid (log would blow up)
    min_p = float(monthly["price_eur_mwh"].min())
    raise ValueError(f"PRICE_SHIFT too small. min(price)={min_p:.2f}. Increase PRICE_SHIFT.")

monthly["y"] = np.log(monthly["price_shifted"])


# =============================================================================
# 2) Anchored mean path + AR(1) w/ monthly seasonality on deviations
# =============================================================================
mask = (monthly["date"] >= pd.Timestamp(ANCHOR_START)) & (monthly["date"] <= pd.Timestamp(ANCHOR_END))
if mask.sum() < 24:
    raise ValueError("Anchor window too short/missing. Adjust ANCHOR_START/ANCHOR_END.")

# Anchor uses the *median* shifted level in the window (more robust than mean)
anchor_date = monthly.loc[mask, "date"].iloc[-1]
anchor_level_shifted = float(monthly.loc[mask, "price_shifted"].median())
mu0 = float(np.log(anchor_level_shifted))

# months_from_anchor is negative for history before anchor and positive after
months_from_anchor = (
    (monthly["date"].dt.year - anchor_date.year) * 12
    + (monthly["date"].dt.month - anchor_date.month)
)

# Deterministic mean path (log space)
monthly["mu"] = mu0 + (months_from_anchor / 12.0) * np.log(1.0 + float(ANNUAL_DRIFT))

# Deviations around the deterministic mean
monthly["z"] = monthly["y"] - monthly["mu"]
monthly["z_lag"] = monthly["z"].shift(1)

# Seasonality: month dummies (month=1 is baseline due to drop_first=True)
monthly["month_num"] = monthly["date"].dt.month.astype(int)
month_dummies = pd.get_dummies(monthly["month_num"], prefix="m", drop_first=True).astype(float)

X = pd.concat([monthly["z_lag"], month_dummies], axis=1)
X = sm.add_constant(X)

# Fit only rows where lag exists (dropna)
reg = pd.concat([monthly["z"], X], axis=1).dropna()
model = sm.OLS(reg["z"].astype(float), reg[X.columns].astype(float)).fit()
print(model.summary())

phi_hat = float(model.params["z_lag"])
const_hat = float(model.params["const"])
season_params = {k: float(v) for k, v in model.params.items() if k.startswith("m_")}

# Safety guard: if phi ~= 1, the deviation process is too close to non-stationary
if abs(phi_hat) >= 0.98:
    raise ValueError(f"Estimated phi={phi_hat:.3f} too close to 1. Model is not safely mean-reverting.")

# Fitted values + residuals
# Note: this is the full in-sample fit (incl. intercept + seasonality)
z_hat_hist = (
    const_hat
    + phi_hat * reg["z_lag"].values
    + np.sum([season_params.get(c, 0.0) * reg[c].values for c in month_dummies.columns], axis=0)
)
eps_hat = (reg["z"].values - z_hat_hist).astype(float)

# ---- Residual stabilization (optional but practical) ----
# 1) Clip extremes (winsorize)
# 2) Scale down overall magnitude (keeps the same shape, smaller amplitude)
eps_sigma = float(np.std(eps_hat, ddof=1)) if len(eps_hat) > 2 else float(np.std(eps_hat))
clip = RESID_CLIP_SIGMA * eps_sigma if eps_sigma > 0 else 0.0
if clip > 0:
    eps_hat = np.clip(eps_hat, -clip, clip)
eps_hat = eps_hat * float(RESID_SCALE)

# Clamp starting deviation (avoid starting from an unusually extreme last point)
z_hist_sigma = float(np.std(monthly["z"].dropna().values, ddof=1))
z0_raw = float(monthly["z"].iloc[-1])
z0_clip = Z0_CLIP_SIGMA * z_hist_sigma if z_hist_sigma > 0 else 0.0
z0 = float(np.clip(z0_raw, -z0_clip, z0_clip)) if z0_clip > 0 else z0_raw

# Simulation phi (optional shrink)
phi_sim = float(phi_hat) * float(PHI_SIM_SHRINK)
phi_sim = float(np.clip(phi_sim, -0.98, 0.98))

print("\n--- Price MC settings ---")
print(f"Anchor: {ANCHOR_START}..{ANCHOR_END} (anchor date {anchor_date.date()})")
print(f"Anchor median (unshifted): {float(monthly.loc[mask,'price_eur_mwh'].median()):.2f} €/MWh")
print(f"phi_hat={phi_hat:.4f} -> phi_sim={phi_sim:.4f}  (PHI_SIM_SHRINK={PHI_SIM_SHRINK})")
print(f"Residual sigma={eps_sigma:.4f}  clip=±{RESID_CLIP_SIGMA}σ  scale={RESID_SCALE}")
print(f"z0_raw={z0_raw:.4f} -> z0_used={z0:.4f}  (Z0_CLIP_SIGMA={Z0_CLIP_SIGMA})")
print(f"Block size months: {BLOCK_SIZE_MONTHS}")
print("Output frequency: MONTHLY (MS)")


def season_effect(month: int) -> float:
    """
    Return the seasonal dummy effect for a given month.
    Month=1 is the baseline (0.0) because we dropped the first dummy column.
    """
    return float(season_params.get(f"m_{month}", 0.0))


# =============================================================================
# 3) Monte Carlo forecast (monthly for 30 years)
# =============================================================================
last_date = monthly["date"].max()
horizon_months = FORECAST_YEARS * 12

future_dates = pd.date_range(
    last_date + pd.offsets.MonthBegin(1),
    periods=horizon_months,
    freq="MS",
)

# Deterministic mean (mu) for future months
months_ahead = (
    (future_dates.year - anchor_date.year) * 12
    + (future_dates.month - anchor_date.month)
).astype(int)

mu_future = mu0 + (months_ahead / 12.0) * np.log(1.0 + float(ANNUAL_DRIFT))
mu_future = mu_future.astype(float)

rng = np.random.default_rng(RANDOM_SEED)

rows = []

for sim in range(1, N_SIMS + 1):
    eps_path = block_bootstrap_1d(
        eps_hat,
        out_len=horizon_months,
        block_size=BLOCK_SIZE_MONTHS,
        rng=rng,
    )

    z_prev = z0

    for i, d in enumerate(future_dates):
        s_t = season_effect(int(d.month))

        # Simulation update:
        # We do NOT propagate the fitted intercept (const_hat) forward.
        # Reason: the intercept often absorbs sample-specific bias; carrying it forward
        # can create artificial long-run drift in z. The deterministic drift is handled by mu_future.
        z_t = phi_sim * z_prev + s_t + float(eps_path[i])

        y_t = float(mu_future[i] + z_t)
        p_t = float(np.exp(y_t) - float(PRICE_SHIFT))

        if WRITE_DEBUG_COLUMNS:
            rows.append((d, sim, p_t, y_t, float(mu_future[i]), z_t))
        else:
            rows.append((d, sim, p_t))

        z_prev = z_t

if WRITE_DEBUG_COLUMNS:
    out = pd.DataFrame(
        rows,
        columns=["date", "sim", "price_eur_mwh", "y_log_shifted", "mu_log_shifted", "z_dev"],
    )
else:
    out = pd.DataFrame(rows, columns=["date", "sim", "price_eur_mwh"])

out.to_csv(OUT_FILE, index=False)
print(f"\n[PRICE] Wrote {len(out):,} rows -> {OUT_FILE}")
