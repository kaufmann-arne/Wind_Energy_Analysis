# revenue.py
#
# Central callable revenue module for power_to_profit.py
# - No hardcoded user inputs (TSO/EEG/COD/strike come from args)
# - Accepts production series (monthly MWh) from profit.py
# - Reads market / capture factor / curtailment / EEG strike from CSVs (paths overridable)
# - Returns monthly DataFrame including 'date' and 'revenue_eur'
#
# NOTE: This version keeps "static script" file lookup (relative to base_dir or cwd).
# You said you will do base_dir/package cleanup later.

import os
import numpy as np
import pandas as pd
from pathlib import Path

# (Plotting only used when run directly)
import matplotlib.pyplot as plt


# Absolute path to the folder that contains this file (…/project/revenue)
MODULE_DIR = Path(__file__).resolve().parent


# ============================================================
# DEFAULT FILE PATHS (relative to base_dir or cwd)
# ============================================================
DEFAULT_MARKET_FILE = "day_ahead_market_prices/market_price_forecast_20y_monthly.csv"
DEFAULT_CF_FILE = "capture_factor_history_forecast/capture_factor_forecast_b2.csv"
DEFAULT_CURTAIL_FILE = "curltailment_rate_per_tso/curtailment_forecast_quarterly_by_tso_2021_2046.csv"
DEFAULT_EEG_FILE = "eeg/eeg_strike_forecast_monthly.csv"

# Column names in those files
MARKET_DATE_COL = "date"
MARKET_PRICE_COL = "price_eur_mwh_forecast"

CF_DATE_COL = "month"
CF_PREF_COLS = ["cf_base", "cf_low", "cf_high", "cf_hist"]

CURTAIL_Q_COL = "quarter"
CURTAIL_TSO_COL = "TSO"
CURTAIL_CR_COL = "cr_hat"

EEG_DATE_COL = "date"
EEG_STRIKE_COL = "eeg_strike_used"

# TSO mapping
TSO_MAP = {
    0: "50Hertz",
    1: "TenneT",
    2: "Amprion",
    3: "TransnetBW",
}

# Clips
CF_CLIP = (0.50, 1.05)
CR_CLIP = (0.00, 0.30)


# ============================================================
# HELPERS
# ============================================================
def to_month_start(s: pd.Series) -> pd.Series:
    """Coerce datetimes and normalize to month-start timestamps."""
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())

def quarter_to_month_starts(q: pd.Period) -> pd.DatetimeIndex:
    """Expand a quarter period into the 3 month-start timestamps."""
    start = q.start_time.to_period("M").to_timestamp()
    return pd.date_range(start, periods=3, freq="MS")

def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}. Found: {list(df.columns)}")

def ffill_asof(series_df: pd.DataFrame, date_col: str, value_col: str, target_dates: pd.Series) -> np.ndarray:
    """As-of backward fill from a dated series onto target_dates."""
    s = series_df[[date_col, value_col]].dropna().sort_values(date_col)
    return pd.merge_asof(
        pd.DataFrame({date_col: target_dates}).sort_values(date_col),
        s,
        on=date_col,
        direction="backward",
    )[value_col].values

def _normalize_mwh_input(mwh_monthly_20y) -> pd.DataFrame:
    """
    Accepts:
      - Series with DatetimeIndex (values = MWh)
      - DataFrame with columns ['date','mwh'] or common alternatives
    Returns: DataFrame with columns ['date','mwh']
    """
    if isinstance(mwh_monthly_20y, pd.Series):
        df = mwh_monthly_20y.rename("mwh").to_frame().reset_index().rename(columns={"index": "date"})
        return df[["date", "mwh"]].copy()

    if isinstance(mwh_monthly_20y, pd.DataFrame):
        df = mwh_monthly_20y.copy()
        if "date" not in df.columns:
            raise ValueError("mwh_monthly_20y DataFrame must contain a 'date' column.")

        if "mwh" not in df.columns:
            for alt in ["mwh_gross", "MWh", "MWh_gross", "mwh_gross_park", "mwh_park"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "mwh"})
                    break

        if "mwh" not in df.columns:
            raise ValueError("mwh_monthly_20y must contain an 'mwh' column (or e.g. mwh_gross).")

        return df[["date", "mwh"]].copy()

    raise TypeError("mwh_monthly_20y must be a pandas Series or DataFrame.")


# ============================================================
# CENTRAL API (called by power_to_profit.py)
# ============================================================
def generate_monthly_revenue(
    *,
    tso_id: int,
    mwh_monthly_20y,
    eeg_on: int,
    manual_eeg_strike: float | None,
    cod_date: str | pd.Timestamp,
    forecast_months: int = 240,
    # optional: where to resolve relative file paths (kept simple for now)
    base_dir: str | None = None,
    # optional file overrides (relative to base_dir/cwd)
    market_file: str = DEFAULT_MARKET_FILE,
    cf_file: str = DEFAULT_CF_FILE,
    curtail_file: str = DEFAULT_CURTAIL_FILE,
    eeg_file: str = DEFAULT_EEG_FILE,
) -> pd.DataFrame:
    """
    Build monthly revenue series for a horizon starting at COD month start.

    Parameters
    ----------
    tso_id : int (0..3)
    mwh_monthly_20y : Series or DataFrame
        Monthly gross park energy output (MWh) supplied by profit.py.
    eeg_on : int (0/1)
    manual_eeg_strike : float | None
        If EEG is on and manual is provided, use it (€/MWh) for full horizon.
    cod_date : str | Timestamp
        Start month (COD). Revenue series begins at this month start.
    forecast_months : int
        Number of months to return from COD (default 240).
    base_dir : str | None
        Base directory to resolve the data files. If None -> current working dir.

    Returns
    -------
    DataFrame (monthly) including:
      - date
      - revenue_eur
      - plus components used for analysis
    """

    # --- validate TSO ---
    tso_name = TSO_MAP.get(int(tso_id))
    if tso_name is None:
        raise ValueError(f"tso_id must be one of {list(TSO_MAP.keys())}")

    # --- normalize COD to month-start ---
    cod = pd.Timestamp(cod_date)
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

    # --- resolve file paths (static script mode) ---
    if base_dir is None:
        base_dir = MODULE_DIR
    else:
        base_dir = Path(base_dir).resolve()

    market_path  = base_dir / market_file
    cf_path      = base_dir / cf_file
    curtail_path = base_dir / curtail_file
    eeg_path     = base_dir / eeg_file


    # --- normalize production input to df(date,mwh) ---
    base = _normalize_mwh_input(mwh_monthly_20y)
    base["date"] = to_month_start(base["date"])
    base["mwh_gross_park"] = pd.to_numeric(base["mwh"], errors="coerce")
    base = base.dropna(subset=["date", "mwh_gross_park"]).copy()
    base = base.sort_values("date").reset_index(drop=True)

    # Keep only rows at/after COD and take exactly forecast_months
    base = base[base["date"] >= cod].copy()
    if base.empty:
        raise ValueError("mwh_monthly_20y has no rows at/after COD.")

    base = base.sort_values("date").reset_index(drop=True)
    if len(base) < int(forecast_months):
        raise ValueError(f"Production series has only {len(base)} months from COD; need {forecast_months} months.")
    base = base.head(int(forecast_months)).copy()

    # Ensure month continuity (optional but helpful)
    expected = pd.date_range(cod, periods=int(forecast_months), freq="MS")
    if not base["date"].equals(pd.Series(expected, name="date")):
        # Try to reindex to full monthly grid (handles missing months)
        base = base.set_index("date").reindex(expected).reset_index().rename(columns={"index": "date"})
        if base["mwh_gross_park"].isna().any():
            missing = base.loc[base["mwh_gross_park"].isna(), "date"].min()
            raise ValueError(
                f"Production series is missing months in the COD horizon. "
                f"First missing month: {missing.date()}. Provide a complete monthly series."
            )

    # ---------------------------------------------------------
    # Load market prices
    # ---------------------------------------------------------
    mkt = pd.read_csv(market_path, parse_dates=[MARKET_DATE_COL])
    mkt[MARKET_DATE_COL] = to_month_start(mkt[MARKET_DATE_COL])
    mkt[MARKET_PRICE_COL] = pd.to_numeric(mkt[MARKET_PRICE_COL], errors="coerce")
    mkt = mkt.dropna(subset=[MARKET_DATE_COL, MARKET_PRICE_COL]).copy()
    mkt = mkt.rename(columns={MARKET_DATE_COL: "date", MARKET_PRICE_COL: "p_market"})

    base = base.merge(mkt, on="date", how="left")
    if base["p_market"].isna().any():
        missing = base.loc[base["p_market"].isna(), "date"].min()
        raise ValueError(f"Market price missing for horizon. First missing month: {missing.date()}")

    # ---------------------------------------------------------
    # Load capture factor (asof fill)
    # ---------------------------------------------------------
    cf = pd.read_csv(cf_path)
    cf[CF_DATE_COL] = to_month_start(cf[CF_DATE_COL])
    cf_col = pick_first_existing_col(cf, CF_PREF_COLS)
    cf[cf_col] = pd.to_numeric(cf[cf_col], errors="coerce")
    cf = cf.rename(columns={CF_DATE_COL: "date", cf_col: "cf_raw"}).sort_values("date")

    base["cf"] = ffill_asof(cf, "date", "cf_raw", base["date"])
    base["cf"] = np.clip(base["cf"], CF_CLIP[0], CF_CLIP[1])

    # ---------------------------------------------------------
    # Load curtailment forecast, select TSO, expand Q->M
    # ---------------------------------------------------------
    curt = pd.read_csv(curtail_path)
    curt[CURTAIL_Q_COL] = pd.PeriodIndex(curt[CURTAIL_Q_COL].astype(str), freq="Q")
    curt[CURTAIL_CR_COL] = pd.to_numeric(curt[CURTAIL_CR_COL], errors="coerce")
    curt = curt.dropna(subset=[CURTAIL_Q_COL, CURTAIL_TSO_COL, CURTAIL_CR_COL]).copy()
    curt = curt[curt[CURTAIL_TSO_COL] == tso_name].copy()
    if curt.empty:
        raise ValueError(f"No curtailment rows found for TSO='{tso_name}' in {curtail_file}")

    rows = []
    for _, r in curt.iterrows():
        q = r[CURTAIL_Q_COL]
        for m in quarter_to_month_starts(q):
            rows.append({"date": pd.Timestamp(m), "cr_raw": float(r[CURTAIL_CR_COL])})

    cr_m = (
        pd.DataFrame(rows)
        .groupby("date", as_index=False)["cr_raw"].mean()
        .sort_values("date")
    )

    base = base.merge(cr_m, on="date", how="left")
    base["cr_raw"] = base["cr_raw"].ffill().bfill()
    base["cr"] = np.clip(base["cr_raw"], CR_CLIP[0], CR_CLIP[1])

    # ---------------------------------------------------------
    # Load EEG strike series (to pick COD strike if manual is None)
    # ---------------------------------------------------------
    eeg = pd.read_csv(eeg_path, parse_dates=[EEG_DATE_COL])
    eeg[EEG_DATE_COL] = to_month_start(eeg[EEG_DATE_COL])
    eeg[EEG_STRIKE_COL] = pd.to_numeric(eeg[EEG_STRIKE_COL], errors="coerce")
    eeg = eeg.dropna(subset=[EEG_DATE_COL, EEG_STRIKE_COL]).copy()
    eeg = eeg.rename(columns={EEG_DATE_COL: "date", EEG_STRIKE_COL: "eeg_strike_series"}).sort_values("date")

    if int(eeg_on) == 1:
        if manual_eeg_strike is not None:
            strike_cod = float(manual_eeg_strike)
            strike_source = "manual"
        else:
            strike_cod_arr = ffill_asof(eeg, "date", "eeg_strike_series", pd.Series([cod]))
            strike_cod = float(strike_cod_arr[0]) if np.isfinite(strike_cod_arr[0]) else np.nan
            if not np.isfinite(strike_cod):
                raise ValueError(
                    "Could not determine EEG strike at/before COD from EEG file. "
                    "Provide manual_eeg_strike or ensure EEG series includes values <= COD."
                )
            strike_source = "from_series_asof"

        base["eeg_on"] = 1
        base["strike_cod_fixed"] = strike_cod
        base["strike_source"] = strike_source
    else:
        base["eeg_on"] = 0
        base["strike_cod_fixed"] = np.nan
        base["strike_source"] = "off"

    # ---------------------------------------------------------
    # Revenue calculation
    # ---------------------------------------------------------
    base["mwh_delivered"] = base["mwh_gross_park"] * (1.0 - base["cr"])
    base["p_wind_merchant"] = base["p_market"] * base["cf"]

    if int(eeg_on) == 1:
        base["p_wind_realised"] = np.maximum(base["p_wind_merchant"], base["strike_cod_fixed"])
    else:
        base["p_wind_realised"] = base["p_wind_merchant"]

    base["revenue_eur"] = base["mwh_delivered"] * base["p_wind_realised"]

    if int(eeg_on) == 1:
        base["eeg_premium_eur_per_mwh"] = np.maximum(0.0, base["strike_cod_fixed"] - base["p_wind_merchant"])
        base["eeg_premium_eur"] = base["eeg_premium_eur_per_mwh"] * base["mwh_delivered"]
    else:
        base["eeg_premium_eur_per_mwh"] = 0.0
        base["eeg_premium_eur"] = 0.0

    return base[
        [
            "date",
            "mwh_gross_park", "cr", "mwh_delivered",
            "p_market", "cf", "p_wind_merchant",
            "eeg_on", "strike_cod_fixed", "p_wind_realised",
            "revenue_eur", "eeg_premium_eur",
            "strike_source",
        ]
    ].copy()


# ============================================================
# Standalone run (debugging)
# ============================================================
if __name__ == "__main__":
    # Example: start today (month start) for 240 months
    cod = pd.Timestamp.today()
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

    dates = pd.date_range(cod, periods=240, freq="MS")
    mock_mwh = pd.Series(12000.0, index=dates)

    out = generate_monthly_revenue(
        tso_id=0,
        mwh_monthly_20y=mock_mwh,
        eeg_on=1,
        manual_eeg_strike=None,
        cod_date=str(cod.date()),
        forecast_months=240,
        base_dir=os.getcwd(),
    )

    out.to_csv("windpark_revenue_forecast_20y_monthly.csv", index=False)
    print("Saved: windpark_revenue_forecast_20y_monthly.csv")

    # Simple chart for sanity
    plot_df = out[["date", "revenue_eur"]].copy()
    plot_df["revenue_rolling_12m"] = plot_df["revenue_eur"].rolling(12, min_periods=1).mean()

    plt.figure()
    plt.plot(plot_df["date"], plot_df["revenue_eur"], label="Monthly revenue")
    plt.plot(plot_df["date"], plot_df["revenue_rolling_12m"], label="Rolling 12M average")
    plt.xlabel("Date")
    plt.ylabel("Revenue (€)")
    plt.title("Wind park revenue forecast (monthly)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("windpark_revenue_over_time.png", dpi=200)
    print("Saved: windpark_revenue_over_time.png")
