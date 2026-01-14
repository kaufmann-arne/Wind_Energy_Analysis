import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- CONFIG ----------------
PRICES_HOURLY = "prices_hourly_merged.csv"
WIND_HOURLY   = "wind_onshore_hourly_merged.csv"
AUCTIONS_CSV  = "eeg_auction_strike_prices.csv"

OUT_MONTHLY_PANEL = "panel_monthly_prices.csv"
OUT_FORECAST_20Y  = "forecast_merchant_vs_eeg_20y.csv"

FORECAST_YEARS = 20
STRIKE_FORWARD = "rolling_24m"   # or "last_known"

# Recommended: start EEG analysis when auctions exist
# Set to "2017-05-01" to avoid NaNs pre-auction, or None to keep all months.
EEG_START_CUTOFF = "2017-05-01"
# ----------------------------------------


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")
    return df


def add_month_dummies(df: pd.DataFrame, month_col="month") -> pd.DataFrame:
    df = df.copy()
    df["month_num"] = df[month_col].dt.month
    d = pd.get_dummies(df["month_num"], prefix="m", drop_first=True)
    return pd.concat([df, d], axis=1)


def normalize_month_series(s: pd.Series) -> pd.Series:
    """
    Ensure a month-start timestamp Series, timezone-naive.
    Safe for Series (no tz_localize on Index required).
    """
    out = pd.to_datetime(s, errors="coerce")
    out = out.dt.to_period("M").dt.to_timestamp()
    # if timezone-aware, drop tz (Series-safe)
    if hasattr(out.dt, "tz"):
        try:
            out = out.dt.tz_localize(None)
        except Exception:
            pass
    return out


def load_and_align_hourly() -> pd.DataFrame:
    p = drop_empty_columns(pd.read_csv(PRICES_HOURLY, parse_dates=["datetime"]))
    w = drop_empty_columns(pd.read_csv(WIND_HOURLY, parse_dates=["datetime"]))

    p = p[["datetime", "price_eur_mwh"]].copy()
    w = w[["datetime", "wind_onshore_mwh"]].copy()

    hw = p.merge(w, on="datetime", how="inner").sort_values("datetime")
    hw = hw.dropna(subset=["price_eur_mwh", "wind_onshore_mwh"])
    return hw


def make_monthly_panel(hw: pd.DataFrame) -> pd.DataFrame:
    hw = hw.copy()
    hw["month"] = hw["datetime"].dt.to_period("M").dt.to_timestamp()

    # Monthly mean market price
    mkt = hw.groupby("month", as_index=False)["price_eur_mwh"].mean()
    mkt = mkt.rename(columns={"price_eur_mwh": "market_price_eur_mwh"})

    # Monthly wind MWh sum
    wind = hw.groupby("month", as_index=False)["wind_onshore_mwh"].sum()
    wind = wind.rename(columns={"wind_onshore_mwh": "wind_mwh"})

    # Monthly wind-weighted capture price: sum(price*wind)/sum(wind)
    hw["rev_weight"] = hw["price_eur_mwh"] * hw["wind_onshore_mwh"]
    wind_rev = hw.groupby("month", as_index=False)["rev_weight"].sum()
    wind_rev = wind_rev.rename(columns={"rev_weight": "wind_revenue_eur"})

    panel = mkt.merge(wind, on="month").merge(wind_rev, on="month")
    panel["wind_capture_price_eur_mwh"] = np.where(
        panel["wind_mwh"] > 0,
        panel["wind_revenue_eur"] / panel["wind_mwh"],
        np.nan
    )

    panel["capture_factor"] = panel["wind_capture_price_eur_mwh"] / panel["market_price_eur_mwh"]

    panel = panel.sort_values("month").reset_index(drop=True)
    panel["t"] = np.arange(len(panel))
    return panel


def fit_capture_factor_model(panel: pd.DataFrame):
    df = add_month_dummies(panel, "month")
    feat = ["t", "wind_mwh"] + [c for c in df.columns if c.startswith("m_")]

    X = df[feat].astype(float)
    y = df["capture_factor"].astype(float)

    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    metrics = {"r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred)}
    return model, feat, pred, metrics


def load_and_prepare_auctions(panel_months: pd.Series) -> pd.Series:
    a = drop_empty_columns(pd.read_csv(AUCTIONS_CSV))

    # parse dd.mm.yyyy
    a["auction_date"] = pd.to_datetime(a["auction_date"], dayfirst=True, errors="coerce")
    a = a.dropna(subset=["auction_date", "strike_eur_mwh"]).sort_values("auction_date")

    # month-start timestamps
    a["month"] = a["auction_date"].dt.to_period("M").dt.to_timestamp()
    a = a.groupby("month", as_index=False)["strike_eur_mwh"].mean().sort_values("month")

    print("\nAuctions parsed:", a["month"].min().date(), "to", a["month"].max().date())
    print("Auction rounds used:", len(a))

    # Normalize panel months the same way
    mi = normalize_month_series(panel_months)
    # Align and ffill
    s = a.set_index("month")["strike_eur_mwh"].reindex(mi).sort_index().ffill()

    # Return as a plain Series aligned to panel order
    s = pd.Series(s.values, index=panel_months.index, name="strike_eur_mwh")
    return s


def main():
    hw = load_and_align_hourly()
    panel = make_monthly_panel(hw)

    # Optional cutoff to avoid pre-auction months (recommended)
    if EEG_START_CUTOFF is not None:
        panel = panel[panel["month"] >= pd.Timestamp(EEG_START_CUTOFF)].copy()
        panel = panel.reset_index(drop=True)
        panel["t"] = np.arange(len(panel))

    # Merchant regression (capture factor)
    cf_model, cf_feat, cf_pred, cf_metrics = fit_capture_factor_model(panel)
    panel["capture_factor_pred"] = cf_pred
    panel["merchant_price_pred_eur_mwh"] = panel["market_price_eur_mwh"] * panel["capture_factor_pred"]

    print("\n=== Merchant model (capture factor regression) ===")
    print("R²:", round(cf_metrics["r2"], 3))
    print("MAE (capture factor):", round(cf_metrics["mae"], 4))

    # Auction strikes + EEG supported price/premium
    panel["strike_eur_mwh"] = load_and_prepare_auctions(panel["month"])

    mask = panel["strike_eur_mwh"].notna()

    panel["eeg_supported_price_pred_eur_mwh"] = np.nan
    panel.loc[mask, "eeg_supported_price_pred_eur_mwh"] = np.maximum(
        panel.loc[mask, "merchant_price_pred_eur_mwh"],
        panel.loc[mask, "strike_eur_mwh"]
    )

    panel["premium_pred_eur_mwh"] = np.nan
    panel.loc[mask, "premium_pred_eur_mwh"] = np.maximum(
        0.0,
        panel.loc[mask, "strike_eur_mwh"] - panel.loc[mask, "merchant_price_pred_eur_mwh"]
    )

    panel_out = panel[[
        "month",
        "market_price_eur_mwh",
        "wind_capture_price_eur_mwh",
        "wind_mwh",
        "capture_factor",
        "capture_factor_pred",
        "merchant_price_pred_eur_mwh",
        "strike_eur_mwh",
        "eeg_supported_price_pred_eur_mwh",
        "premium_pred_eur_mwh"
    ]].copy()

    panel_out.to_csv(OUT_MONTHLY_PANEL, index=False)
    print(f"\nWrote monthly panel -> {OUT_MONTHLY_PANEL}")

    # -------- Forecast 20 years (monthly) --------
    last_month = panel["month"].max()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=FORECAST_YEARS * 12, freq="MS")
    future = pd.DataFrame({"month": future_months}).sort_values("month").reset_index(drop=True)

    # Simple baseline scenario (edit later):
    # market price = last 36-month mean
    base_mkt = float(panel["market_price_eur_mwh"].tail(36).mean())
    future["market_price_eur_mwh"] = base_mkt

    # wind volume = month-of-year mean
    tmp = panel.copy()
    tmp["month_num"] = tmp["month"].dt.month
    moy_wind = tmp.groupby("month_num")["wind_mwh"].mean().to_dict()
    future["wind_mwh"] = future["month"].dt.month.map(moy_wind)

    # time index continues
    future["t"] = np.arange(len(panel), len(panel) + len(future))

    # dummies consistent with training
    future = add_month_dummies(future, "month")
    for c in [c for c in cf_feat if c.startswith("m_")]:
        if c not in future.columns:
            future[c] = 0

    # capture + merchant forecast
    future["capture_factor_pred"] = cf_model.predict(future[cf_feat].astype(float))
    future["merchant_price_pred_eur_mwh"] = future["market_price_eur_mwh"] * future["capture_factor_pred"]

    # strike forward (always filled)
    strike_hist = panel["strike_eur_mwh"].dropna()
    if strike_hist.empty:
        raise ValueError("No strike prices in panel after merge. Something is wrong with auction join.")

    if STRIKE_FORWARD == "last_known":
        strike_fwd = float(strike_hist.iloc[-1])
    elif STRIKE_FORWARD == "rolling_24m":
        strike_fwd = float(strike_hist.tail(24).mean())
    else:
        raise ValueError("Unknown STRIKE_FORWARD. Use 'last_known' or 'rolling_24m'.")

    future["strike_eur_mwh"] = strike_fwd
    future["eeg_supported_price_pred_eur_mwh"] = np.maximum(
        future["merchant_price_pred_eur_mwh"], future["strike_eur_mwh"]
    )
    future["premium_pred_eur_mwh"] = np.maximum(
        0.0, future["strike_eur_mwh"] - future["merchant_price_pred_eur_mwh"]
    )

    future_out = future[[
        "month",
        "market_price_eur_mwh",
        "merchant_price_pred_eur_mwh",
        "strike_eur_mwh",
        "eeg_supported_price_pred_eur_mwh",
        "premium_pred_eur_mwh",
        "capture_factor_pred",
        "wind_mwh"
    ]].copy()

    future_out.to_csv(OUT_FORECAST_20Y, index=False)
    print(f"Wrote 20-year forecast -> {OUT_FORECAST_20Y}")


if __name__ == "__main__":
    main()
