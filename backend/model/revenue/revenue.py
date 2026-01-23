"""
One-step Monte Carlo revenue runner + aggregation (server-friendly).

Key design:
- The core function returns DataFrames in-memory (FAST, no disk I/O by default).
- CSV files are written ONLY if write_outputs=True (typically when run manually via CLI).

Inputs:
- mc_inputs_30y_monthly_merged.csv (monthly x sim), produced by your merge script.
  Expected columns:
    date, sim, price_eur_mwh, capture_factor, eeg_strike_eur_mwh,
    cr_tennet, cr_50hertz, cr_amprion, cr_transnetbw

Outputs (returned as DataFrames):
- annual_by_sim:   (sim, project_year) annual sums/means
- annual_percentiles: percentiles across sims by project_year
- totals_by_sim:   project totals per sim
- (optional) monthly_by_sim: big monthly table

If write_outputs=True:
- revenue_mc_annual_by_sim.csv
- revenue_mc_annual_percentiles.csv
- revenue_mc_totals_by_sim.csv
- (optional) revenue_mc_monthly_by_sim.csv  (big)

Usage from another script (no files written):
    from revenue import run_revenue_mc_one_step
    results = run_revenue_mc_one_step(..., write_outputs=False)

Usage from command line (writes files):
    python revenue.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# =============================
# CONFIG (hardcoded paths)
# =============================
REVENUE_DIR = Path(__file__).resolve().parent
MC_INPUTS_PATH = REVENUE_DIR / "mc_inputs_30y_monthly_merged.csv"

TSO_MAP = {0: "50Hertz", 1: "TenneT", 2: "Amprion", 3: "TransnetBW"}
CR_COL_BY_TSO = {
    "TenneT": "cr_tennet",
    "50Hertz": "cr_50hertz",
    "Amprion": "cr_amprion",
    "TransnetBW": "cr_transnetbw",
}

# Small outputs
OUT_ANNUAL_BY_SIM = "revenue_mc_annual_by_sim.csv"
OUT_ANNUAL_PCTLS = "revenue_mc_annual_percentiles.csv"
OUT_TOTALS_BY_SIM = "revenue_mc_totals_by_sim.csv"

# Optional (big)
OUT_MONTHLY = "revenue_mc_monthly_by_sim.csv"

# Percentiles to report
PCTS = [0.10, 0.25, 0.50, 0.75, 0.90]

# Clips (keep consistent with revenue.py)
CF_CLIP = (0.50, 1.05)
CR_CLIP = (0.00, 0.30)


# =============================
# HELPERS
# =============================
def to_month_start(s: pd.Series) -> pd.Series:
    """Normalize to month-start timestamps (MS)."""
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())


def normalize_mwh_input(mwh_monthly) -> pd.DataFrame:
    """
    Accepts:
      - Series with DatetimeIndex (values = MWh)
      - DataFrame with columns ['date','mwh'] or common alternatives
    Returns: DataFrame with columns ['date','mwh_gross_park'] normalized to month-start.
    """
    if isinstance(mwh_monthly, pd.Series):
        df = mwh_monthly.rename("mwh_gross_park").to_frame().reset_index().rename(columns={"index": "date"})
        df["date"] = to_month_start(df["date"])
        df["mwh_gross_park"] = pd.to_numeric(df["mwh_gross_park"], errors="coerce")
        df = df.dropna(subset=["date", "mwh_gross_park"]).copy()
        return df[["date", "mwh_gross_park"]].copy()

    if isinstance(mwh_monthly, pd.DataFrame):
        df = mwh_monthly.copy()
        if "date" not in df.columns:
            raise ValueError("Production DataFrame must have a 'date' column.")

        if "mwh" in df.columns:
            df = df.rename(columns={"mwh": "mwh_gross_park"})
        else:
            for alt in ["mwh_gross", "MWh", "MWh_gross", "mwh_gross_park", "mwh_park"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "mwh_gross_park"})
                    break

        if "mwh_gross_park" not in df.columns:
            raise ValueError("Production must have 'mwh' or an alternative like 'mwh_gross_park'.")

        df["date"] = to_month_start(df["date"])
        df["mwh_gross_park"] = pd.to_numeric(df["mwh_gross_park"], errors="coerce")
        df = df.dropna(subset=["date", "mwh_gross_park"]).copy()
        return df[["date", "mwh_gross_park"]].copy()

    raise TypeError("Production must be a pandas Series or DataFrame.")


def require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where} missing columns: {missing}. Found: {list(df.columns)}")


def annual_percentiles(annual_df: pd.DataFrame, value_col: str, probs: list[float]) -> pd.DataFrame:
    """For each project_year, compute percentiles of value_col across sims."""
    def q(s: pd.Series) -> pd.Series:
        return pd.Series({f"p{int(p*100):02d}": float(s.quantile(p)) for p in probs})

    return (
        annual_df.groupby("project_year")[value_col]
                 .apply(q)
                 .reset_index()
    )


# =============================
# MAIN ONE-STEP FUNCTION
# =============================
def run_revenue_mc_one_step(
    *,
    tso_id: int,
    mwh_monthly,
    eeg_on: int = 1,
    cod_date: str | pd.Timestamp = "2026-01",  # YYYY-MM
    forecast_months: int = 240,                # 20 years
    # server-friendly switches
    write_outputs: bool = False,               # write small CSV outputs
    write_monthly: bool = False,               # include big monthly output in returned dict
) -> Dict[str, pd.DataFrame]:
    """
    Compute MC revenue and annual summaries in one step.

    EEG logic:
    - Strike is taken ONCE at COD month (per simulation) and then held constant.
    - Each month, realised price = max(merchant price, fixed strike) when EEG is on.

    MC input path is hardcoded to:
      backend/model/revenue/mc_inputs_30y_monthly_merged.csv
    """
    tso_name = TSO_MAP.get(int(tso_id))
    if tso_name is None:
        raise ValueError(f"tso_id must be one of {list(TSO_MAP.keys())}")

    cr_col = CR_COL_BY_TSO[tso_name]

    # Normalize COD and build expected month grid
    cod = pd.to_datetime(cod_date, errors="raise")
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)
    expected_dates = pd.date_range(cod, periods=int(forecast_months), freq="MS")

    # Hardcoded MC inputs path
    mc_path = MC_INPUTS_PATH
    if not mc_path.exists():
        raise FileNotFoundError(
            f"MC inputs file not found at: {mc_path}. "
            f"Expected it in: {REVENUE_DIR}"
        )

    # -------------------------
    # Load MC inputs (monthly x sim)
    # -------------------------
    mc = pd.read_csv(mc_path, parse_dates=["date"])
    mc["date"] = to_month_start(mc["date"])
    mc["sim"] = pd.to_numeric(mc["sim"], errors="coerce").astype("Int64")
    mc = mc.dropna(subset=["date", "sim"]).copy()
    mc["sim"] = mc["sim"].astype(int)

    # Validate required columns
    required_cols = ["date", "sim", "price_eur_mwh", "capture_factor", "eeg_strike_eur_mwh", cr_col]
    require_cols(mc, required_cols, where=f"MC inputs file '{mc_path.name}'")

    # Coerce numeric columns
    for c in ["price_eur_mwh", "capture_factor", "eeg_strike_eur_mwh", cr_col]:
        mc[c] = pd.to_numeric(mc[c], errors="coerce")

    # We require market price, CF, and CR. Strike must exist as a column; values are only required when EEG is on.
    mc = mc.dropna(subset=["price_eur_mwh", "capture_factor", cr_col]).copy()

    # Slice to COD horizon (keeps compute tight)
    mc = mc[(mc["date"] >= cod) & (mc["date"] <= expected_dates[-1])].copy()
    if mc.empty:
        raise ValueError(f"MC inputs contain no rows in horizon {cod.date()}..{expected_dates[-1].date()}.")

    # -------------------------
    # Load/prepare production (monthly)
    # -------------------------
    prod = normalize_mwh_input(mwh_monthly)
    prod = prod[prod["date"] >= cod].copy().sort_values("date").reset_index(drop=True)

    if len(prod) < int(forecast_months):
        raise ValueError(f"Production has {len(prod)} months from COD; need {forecast_months}.")

    prod = prod.head(int(forecast_months)).copy()

    # Ensure exact month grid (no missing months)
    if not prod["date"].equals(pd.Series(expected_dates, name="date")):
        prod = prod.set_index("date").reindex(expected_dates).reset_index().rename(columns={"index": "date"})
        if prod["mwh_gross_park"].isna().any():
            miss = prod.loc[prod["mwh_gross_park"].isna(), "date"].min()
            raise ValueError(f"Production missing month {miss.date()} in COD horizon.")

    # -------------------------
    # Merge production onto MC (replicate production across sims)
    # -------------------------
    df = mc.merge(prod, on="date", how="left")
    if df["mwh_gross_park"].isna().any():
        miss = df.loc[df["mwh_gross_park"].isna(), "date"].min()
        raise ValueError(f"Production merge failed (missing) at {miss.date()}.")

    # -------------------------
    # Prepare drivers and clips
    # -------------------------
    df["p_market"] = df["price_eur_mwh"].astype(float)
    df["cf"] = np.clip(df["capture_factor"].astype(float), CF_CLIP[0], CF_CLIP[1])
    df["cr"] = np.clip(df[cr_col].astype(float), CR_CLIP[0], CR_CLIP[1])

    # -------------------------
    # EEG strike selection (fixed at COD per sim)
    # -------------------------
    if int(eeg_on) == 1:
        # Ensure ordered so "first" = COD month per sim (since we sliced to >= cod)
        df = df.sort_values(["sim", "date"]).copy()

        strike_at_cod = df.groupby("sim")["eeg_strike_eur_mwh"].transform("first")
        strike_at_cod = pd.to_numeric(strike_at_cod, errors="coerce")

        if strike_at_cod.isna().any():
            bad_sims = df.loc[strike_at_cod.isna(), "sim"].unique()[:10]
            raise ValueError(
                f"EEG strike missing at COD month for sims {list(bad_sims)} "
                f"(check 'eeg_strike_eur_mwh' for the COD row in MC inputs)."
            )

        df["strike_used"] = strike_at_cod.astype(float)
        df["strike_source"] = "cod_fixed"
        df["eeg_on"] = 1
        # Sanity: strike must be constant within each sim (fixed at COD)
        nuniq = df.groupby("sim")["strike_used"].nunique()
        bad = nuniq[nuniq != 1]
        if not bad.empty:
            raise ValueError(f"BUG: strike_used not constant for sims: {bad.index.tolist()[:10]}")

    else:
        df["strike_used"] = np.nan
        df["strike_source"] = "off"
        df["eeg_on"] = 0

    # -------------------------
    # Revenue math (vectorized)
    # -------------------------
    df["mwh_delivered"] = df["mwh_gross_park"].astype(float) * (1.0 - df["cr"])
    df["p_wind_merchant"] = df["p_market"] * df["cf"]

    if int(eeg_on) == 1:
        df["p_wind_realised"] = np.maximum(df["p_wind_merchant"], df["strike_used"].astype(float))
        df["eeg_premium_eur_per_mwh"] = np.maximum(0.0, df["strike_used"].astype(float) - df["p_wind_merchant"])
        df["eeg_premium_eur"] = df["eeg_premium_eur_per_mwh"] * df["mwh_delivered"]
    else:
        df["p_wind_realised"] = df["p_wind_merchant"]
        df["eeg_premium_eur_per_mwh"] = 0.0
        df["eeg_premium_eur"] = 0.0

    df["revenue_eur"] = df["mwh_delivered"] * df["p_wind_realised"]

    # -------------------------
    # Project year index (1..N) based on COD
    # -------------------------
    months_from_cod = (df["date"].dt.year - cod.year) * 12 + (df["date"].dt.month - cod.month)
    df["project_year"] = (months_from_cod // 12 + 1).astype(int)

    # =========================
    # AGGREGATE OUTPUTS (small)
    # =========================
    annual = (
        df.groupby(["sim", "project_year"], as_index=False)
          .agg(
              revenue_eur=("revenue_eur", "sum"),
              eeg_premium_eur=("eeg_premium_eur", "sum"),
              mwh_delivered=("mwh_delivered", "sum"),
              mwh_gross_park=("mwh_gross_park", "sum"),
              avg_p_market=("p_market", "mean"),
              avg_p_realised=("p_wind_realised", "mean"),
              avg_cf=("cf", "mean"),
              avg_cr=("cr", "mean"),
          )
    )

    pctls = annual_percentiles(annual, "revenue_eur", PCTS)

    totals = (
        annual.groupby("sim", as_index=False)
              .agg(
                  revenue_total_eur=("revenue_eur", "sum"),
                  eeg_premium_total_eur=("eeg_premium_eur", "sum"),
                  mwh_delivered_total=("mwh_delivered", "sum"),
                  mwh_gross_park_total=("mwh_gross_park", "sum"),
                  avg_p_market=("avg_p_market", "mean"),
                  avg_p_realised=("avg_p_realised", "mean"),
              )
    )

    results: Dict[str, pd.DataFrame] = {
        "annual_by_sim": annual,
        "annual_percentiles": pctls,
        "totals_by_sim": totals,
    }

    # Optional monthly output (big; keep minimal set of columns)
    if write_monthly:
        keep_cols = [
            "date", "sim", "project_year",
            "mwh_gross_park", "cr", "mwh_delivered",
            "p_market", "cf", "p_wind_merchant",
            "eeg_on", "strike_used", "p_wind_realised",
            "revenue_eur", "eeg_premium_eur",
            "strike_source",
        ]
        results["monthly_by_sim"] = df[keep_cols].sort_values(["sim", "date"]).reset_index(drop=True)

    # =========================
    # EEG meta for UI
    # =========================
    if int(eeg_on) == 1:
        # strike_used is constant per sim; take one value per sim
        strike_by_sim = (
            df.groupby("sim", as_index=False)["strike_used"]
              .first()
              .rename(columns={"strike_used": "eeg_strike_eur_per_mwh"})
        )
        results["eeg_meta"] = strike_by_sim
    else:
        results["eeg_meta"] = pd.DataFrame({"sim": totals["sim"], "eeg_strike_eur_per_mwh": np.nan})


    # =========================
    # WRITE OUTPUTS (optional)
    # =========================
    if write_outputs:
        out_dir = REVENUE_DIR
        annual.to_csv(out_dir / OUT_ANNUAL_BY_SIM, index=False)
        pctls.to_csv(out_dir / OUT_ANNUAL_PCTLS, index=False)
        totals.to_csv(out_dir / OUT_TOTALS_BY_SIM, index=False)

        if write_monthly:
            results["monthly_by_sim"].to_csv(out_dir / OUT_MONTHLY, index=False)


    return results


# =============================
# Standalone example (CLI)
# =============================
if __name__ == "__main__":
    cod = pd.Timestamp("2026-01-01")
    dates = pd.date_range(cod, periods=240, freq="MS")
    mock_prod = pd.Series(12000.0, index=dates)

    res = run_revenue_mc_one_step(
        tso_id=1,  # TenneT
        mwh_monthly=mock_prod,
        eeg_on=1,
        cod_date="2026-01",  # YYYY-MM
        forecast_months=240,
        write_outputs=False,
        write_monthly=False,
    )

    print("[DONE] annual_by_sim rows:", len(res["annual_by_sim"]))
    print("[DONE] totals_by_sim rows:", len(res["totals_by_sim"]))
