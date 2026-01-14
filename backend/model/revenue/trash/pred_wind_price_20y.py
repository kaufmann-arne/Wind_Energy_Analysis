import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "smard_calc_wind_onshore_monthly.csv"
DATE_COL = "date"
PRICE_COL = "wind_onshore_eur_mwh_calc"

FORECAST_YEARS = 20

ANCHOR_START = "2020-01-01"
ANCHOR_END   = "2024-12-01"

DRIFT_SCENARIOS = {
    "low":  0.03,
    "base": 0.05,
    "high": 0.07,
}

# Seasonality damping (0=no season, 1=full fitted season)
SEASON_SHRINK = 0.25  # lowered from 0.35 to reduce season strength

# Fourier seasonality complexity: 1 = annual wave; 2 adds semi-annual wave
FOURIER_K = 1  # start with 1; set to 2 if you want slightly richer seasonality

# Add randomness to the seasonal pattern itself (log-space). 0 = none.
# Think of this as "winter isn't always exactly winter" year-to-year.
SEASON_JITTER_STD = 0.03  # try 0.02–0.06; set 0 to disable

N_PATHS = 500
SEED = 42

# Overall volatility multiplier
VOL_MULT = 1.20  # increased from 1.00 to add more randomness

# If True, use month-dependent sigma estimated from history (recommended)
USE_MONTHLY_SIGMA = True

OUTPUT_CSV = "forecast_market_price_ar1_stochastic_3scen.csv"
OUTPUT_PNG = "forecast_market_price_ar1_stochastic_3scen.png"

# -----------------------------
def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def fit_fourier_seasonality(log_price: pd.Series, dates: pd.Series, k: int) -> dict[int, float]:
    """
    Fit smooth month-of-year seasonality in log-space using Fourier terms:
      lp_t = c + sum_{j=1..k} [a_j sin(2π j m/12) + b_j cos(2π j m/12)] + error
    Returns month->seasonality value with zero-mean across months.
    """
    df = pd.DataFrame({"date": dates, "lp": log_price})
    m = df["date"].dt.month.values.astype(float)

    # Design matrix: constant + sin/cos terms
    X = [np.ones_like(m)]
    for j in range(1, k + 1):
        X.append(np.sin(2.0 * np.pi * j * m / 12.0))
        X.append(np.cos(2.0 * np.pi * j * m / 12.0))
    X = np.column_stack(X)

    # OLS
    beta, *_ = np.linalg.lstsq(X, df["lp"].values, rcond=None)

    # Build month seasonal component excluding intercept (beta[0])
    seas = {}
    months = np.arange(1, 13, dtype=float)
    Xm = [np.ones_like(months)]
    for j in range(1, k + 1):
        Xm.append(np.sin(2.0 * np.pi * j * months / 12.0))
        Xm.append(np.cos(2.0 * np.pi * j * months / 12.0))
    Xm = np.column_stack(Xm)

    lp_hat = Xm @ beta
    # seasonality is the fitted deviation from mean across months
    s = lp_hat - lp_hat.mean()
    for i, mm in enumerate(range(1, 13)):
        seas[int(mm)] = float(s[i])

    return seas

def make_mu_log(dates: pd.Series, anchor_level: float, anchor_date: pd.Timestamp, g_annual: float) -> np.ndarray:
    months_from_anchor = (dates.dt.year - anchor_date.year) * 12 + (dates.dt.month - anchor_date.month)
    return np.log(anchor_level) + (months_from_anchor / 12.0) * np.log(1.0 + g_annual)

def fit_ar1_on_residuals(y: np.ndarray) -> tuple[float, float]:
    y = pd.Series(y).dropna()
    y_lag = y.shift(1).dropna()
    y_now = y.loc[y_lag.index]

    denom = np.sum(y_lag.values**2)
    phi = float(np.sum(y_now.values * y_lag.values) / denom) if denom != 0 else 0.0

    u = y_now.values - phi * y_lag.values
    sigma = float(np.std(u, ddof=1)) if len(u) > 2 else 0.0
    return phi, sigma

def estimate_monthly_sigma(resid: np.ndarray, dates: pd.Series) -> dict[int, float]:
    """
    Estimate month-specific shock volatility (sigma_u) in log-space from residuals.
    This creates more realistic seasonal volatility (often higher in winter).
    """
    tmp = pd.DataFrame({"date": dates, "resid": resid})
    tmp["m"] = tmp["date"].dt.month
    sig = tmp.groupby("m")["resid"].std(ddof=1).to_dict()

    # fallback if some months missing
    overall = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0
    for m in range(1, 13):
        sig.setdefault(m, overall)
        if not np.isfinite(sig[m]) or sig[m] == 0.0:
            sig[m] = overall
    return {int(k): float(v) for k, v in sig.items()}

def simulate_paths(
    mu_path: np.ndarray,
    seas_future: np.ndarray,
    future_months: np.ndarray,
    phi: float,
    sigma_global: float,
    sigma_by_month: dict[int, float] | None,
    yT: float,
    n_paths: int,
    rng: np.random.Generator,
    vol_mult: float,
    season_jitter_std: float
) -> np.ndarray:
    """
    AR(1) around drifting mean in deseasonalized log-space:
      y_t = mu_t + phi*(y_{t-1} - mu_{t-1}) + eps_t
    Then add back (damped) seasonality + optional seasonal jitter.
    Returns simulated log price (n_paths, T)
    """
    T = len(mu_path) - 1
    sims = np.zeros((n_paths, T), dtype=float)

    y_prev = np.full(n_paths, yT, dtype=float)

    # Optional: one seasonal jitter draw per path per month (small)
    # This prevents a perfectly repeating seasonal pattern.
    if season_jitter_std > 0:
        season_jitter = rng.normal(0.0, season_jitter_std, size=(n_paths, T))
    else:
        season_jitter = 0.0

    for t in range(1, len(mu_path)):
        mu_prev = mu_path[t - 1]
        mu_now  = mu_path[t]

        m = int(future_months[t - 1])
        sigma_t = sigma_global
        if sigma_by_month is not None:
            sigma_t = sigma_by_month.get(m, sigma_global)

        eps = rng.normal(0.0, sigma_t * vol_mult, size=n_paths)

        y_now = mu_now + phi * (y_prev - mu_prev) + eps

        seas = seas_future[t - 1]
        sims[:, t - 1] = y_now + seas + (season_jitter[:, t - 1] if isinstance(season_jitter, np.ndarray) else 0.0)

        y_prev = y_now

    return sims

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, INPUT_FILE), parse_dates=[DATE_COL])

    df[DATE_COL] = df[DATE_COL].apply(month_start)
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL, keep="last").reset_index(drop=True)

    if PRICE_COL not in df.columns:
        raise ValueError(f"Column '{PRICE_COL}' not found. Columns: {list(df.columns)}")

    df = df.dropna(subset=[PRICE_COL])
    if (df[PRICE_COL] <= 0).any():
        raise ValueError("Non-positive monthly prices found; log model needs >0.")

    df["log_price"] = np.log(df[PRICE_COL].astype(float))

    # Anchor window
    mask_anchor = (df[DATE_COL] >= pd.Timestamp(ANCHOR_START)) & (df[DATE_COL] <= pd.Timestamp(ANCHOR_END))
    if mask_anchor.sum() < 12:
        raise ValueError("Anchor window too short or missing. Adjust ANCHOR_START/END.")

    anchor_level = float(df.loc[mask_anchor, PRICE_COL].median())
    anchor_date = month_start(df.loc[mask_anchor, DATE_COL].iloc[-1])

    # ---- NEW: Smooth Fourier seasonality (then damp)
    season = fit_fourier_seasonality(df["log_price"], df[DATE_COL], k=FOURIER_K)
    df["log_season"] = df[DATE_COL].dt.month.map(season).astype(float) * SEASON_SHRINK

    # Use BASE scenario drift for fitting phi/sigma
    base_g = DRIFT_SCENARIOS["base"]
    mu_hist = make_mu_log(df[DATE_COL], anchor_level, anchor_date, base_g)

    y_hist_deseas = (df["log_price"].values - df["log_season"].values)
    resid = y_hist_deseas - mu_hist

    phi, sigma_global = fit_ar1_on_residuals(resid)
    phi = max(min(phi, 0.98), 0.0)

    # ---- NEW: month-dependent sigma (optional)
    sigma_by_month = None
    if USE_MONTHLY_SIGMA:
        sigma_by_month = estimate_monthly_sigma(resid, df[DATE_COL])

    print(f"Anchor level (median {ANCHOR_START}..{ANCHOR_END}): {anchor_level:.2f} €/MWh (anchor date {anchor_date.date()})")
    print(f"Seasonality: Fourier k={FOURIER_K}, shrink={SEASON_SHRINK:.2f}, jitter_std={SEASON_JITTER_STD:.3f}")
    print(f"AR(1): phi={phi:.4f}, sigma_global={sigma_global:.4f}, monthly_sigma={USE_MONTHLY_SIGMA}, VOL_MULT={VOL_MULT:.2f}")

    # Forecast dates
    last_date = df[DATE_COL].max()
    forecast_months = FORECAST_YEARS * 12
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=forecast_months, freq="MS")
    future_months_arr = np.array([d.month for d in future_dates], dtype=int)

    yT = float(y_hist_deseas[-1])

    out = pd.DataFrame({DATE_COL: pd.concat([df[DATE_COL], pd.Series(future_dates)], ignore_index=True)})
    out["price_hist_eur_mwh"] = pd.concat([df[PRICE_COL], pd.Series([np.nan] * len(future_dates))], ignore_index=True)

    # Damped seasonality for future
    seas_future = np.array([season[d.month] for d in future_dates], dtype=float) * SEASON_SHRINK

    # Build scenario forecasts with uncertainty bands
    for scen, g in DRIFT_SCENARIOS.items():
        # Use a scenario-specific RNG so scenarios aren't artificially identical
        scen_seed = SEED + (0 if scen == "base" else (1 if scen == "low" else 2))
        rng = np.random.default_rng(scen_seed)

        all_dates = pd.Series([last_date] + list(future_dates))
        mu_path = make_mu_log(all_dates, anchor_level, anchor_date, g)

        sims_log = simulate_paths(
            mu_path=mu_path,
            seas_future=seas_future,
            future_months=future_months_arr,
            phi=phi,
            sigma_global=sigma_global,
            sigma_by_month=sigma_by_month,
            yT=yT,
            n_paths=N_PATHS,
            rng=rng,
            vol_mult=VOL_MULT,
            season_jitter_std=SEASON_JITTER_STD
        )

        sims_price = np.exp(sims_log)
        p10 = np.percentile(sims_price, 10, axis=0)
        p50 = np.percentile(sims_price, 50, axis=0)
        p90 = np.percentile(sims_price, 90, axis=0)

        col_p10 = f"price_{scen}_p10"
        col_p50 = f"price_{scen}_p50"
        col_p90 = f"price_{scen}_p90"

        out[col_p10] = np.nan
        out[col_p50] = np.nan
        out[col_p90] = np.nan

        future_mask = out[DATE_COL].isin(future_dates)
        out.loc[future_mask, col_p10] = p10
        out.loc[future_mask, col_p50] = p50
        out.loc[future_mask, col_p90] = p90

    # Save CSV
    out_path = os.path.join(base_dir, OUTPUT_CSV)
    out.to_csv(out_path, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    # Plot
    plt.figure()

    plt.plot(out[DATE_COL], out["price_hist_eur_mwh"], label="History (monthly)")

    plt.plot(out[DATE_COL], out["price_base_p50"], label="Forecast base (P50)")


    plt.plot(out[DATE_COL], out["price_low_p50"], label="Forecast low (P50)")
    plt.plot(out[DATE_COL], out["price_high_p50"], label="Forecast high (P50)")

    plt.xlabel("Date")
    plt.ylabel("€/MWh")
    plt.title("Market price forecast: anchored stochastic AR(1) + drift (monthly)")
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(base_dir, OUTPUT_PNG)
    plt.savefig(png_path, dpi=200)
    print(f"Saved plot: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
