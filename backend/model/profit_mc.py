#
# Orchestrates: revenue (Monte Carlo) + capex + opex + finance => yearly profit, NPV, IRR (per simulation)
#
# Key points:
# - Revenue is produced via run_revenue_mc_one_step() which reads the merged MC inputs CSV
#   (now hardcoded inside revenue2.py) and returns annual_by_sim, totals_by_sim, (optional) monthly_by_sim.
# - CAPEX/OPEX/Finance are deterministic and applied to every simulation.
# - We compute:
#     * yearly_df_by_sim  : (sim, project_year) panel with revenue, opex, debt, profit
#     * sim_summary_df    : one row per sim with NPV/IRR and totals
#     * stats_df          : rich distribution diagnostics (variance, tails, skew/kurtosis, probs)
#     * final_summary_df  : compact "one final result" (mean, P10, P50, P90 + key probabilities)
#
# Server-friendly:
# - No CSVs written by default; all results are returned in-memory.
# - If you want to write CSVs, do it in __main__ or your calling code.

from __future__ import annotations

import numpy as np
import pandas as pd
import numpy_financial as npf
import sys
from pathlib import Path

# NEW: distribution shape diagnostics
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parent

# Add module folders to Python import path
sys.path.insert(0, str(ROOT / "revenue"))              # contains revenue runner file (revenue2.py)
sys.path.insert(0, str(ROOT / "costs" / "capex"))
sys.path.insert(0, str(ROOT / "costs" / "opex"))
sys.path.insert(0, str(ROOT / "costs" / "finance"))

# Revenue runner (MC inputs path is hardcoded in revenue2.py)
from revenue import run_revenue_mc_one_step

import capex as capex_mod
import opex as opex_mod
import finance as finance_mod


# =========================
# CONFIG
# =========================
FORECAST_MONTHS = 240  # 30 years
TERMINAL_VALUE_SHARE = 0.30  # salvage/repowering fraction of CAPEX at end of horizon


# =========================
# Helpers
# =========================
def _compute_npv(cashflows: list[float], discount_rate: float) -> float:
    """NPV with cashflows indexed at t=0..N (t=0 is equity outflow)."""
    return float(sum(cf / ((1.0 + discount_rate) ** t) for t, cf in enumerate(cashflows)))


def _safe_irr(cashflows: list[float]) -> float:
    """Robust IRR. Returns NaN if no sign change or if solver fails."""
    if all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
        return float("nan")
    try:
        return float(npf.irr(cashflows))
    except Exception:
        return float("nan")


def _normalize_mwh_input(mwh_monthly_20y) -> pd.DataFrame:
    """
    Accept production in either of:
      - Series with DatetimeIndex
      - DataFrame with columns ['date','mwh'] or common alternatives
    Return a DataFrame with columns ['date','mwh'].

    Note: run_revenue_mc_one_step() will normalize again internally; this is mainly for validation.
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


def stats_table(series: pd.Series, name: str, *, wacc: float | None = None) -> pd.DataFrame:
    """
    Build a rich 1-row stats table for a numeric series.

    Includes:
    - mean / std / var
    - tail percentiles (P1..P99)
    - skewness and excess kurtosis (fat tails)
    - probabilities:
        * Prob(value < 0)
        * Prob(IRR < WACC) (if metric is IRR and wacc is provided)
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame([{"metric": name}])

    pct_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    pct = {f"p{int(p*100):02d}": float(s.quantile(p)) for p in pct_levels}

    row = {
        "metric": name,
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "var": float(s.var(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        # shape diagnostics
        "skew": float(skew(s.values, bias=False)),
        "kurtosis_excess": float(kurtosis(s.values, fisher=True, bias=False)),
        # headline risk probability
        "prob_lt_0": float((s < 0).mean()),
    }

    # Optional: risk probability relative to WACC for IRR
    if wacc is not None and name.lower().startswith("irr"):
        row["prob_irr_lt_wacc"] = float((s < wacc).mean())

    row.update(pct)
    return pd.DataFrame([row])


def scenario_summary(sim_summary: pd.DataFrame, *, wacc: float) -> pd.DataFrame:
    """
    One compact "final result" table across all simulations:
    mean / P10 / P50 / P90 for key outputs + a couple of key probabilities.

    This is the table you typically display in an app/report.
    """
    def q(col: str, p: float) -> float:
        s = pd.to_numeric(sim_summary[col], errors="coerce").dropna()
        return float(s.quantile(p)) if not s.empty else float("nan")

    def mean(col: str) -> float:
        s = pd.to_numeric(sim_summary[col], errors="coerce").dropna()
        return float(s.mean()) if not s.empty else float("nan")

    metrics = [
        ("npv_eur", "NPV (€)"),
        ("irr", "IRR"),
    ]

    # Add these if present (depends on merges/what you store)
    if "revenue_total_eur" in sim_summary.columns:
        metrics.append(("revenue_total_eur", "Total revenue (€)"))
    if "total_profit_after_opex_and_debt_eur" in sim_summary.columns:
        metrics.append(("total_profit_after_opex_and_debt_eur", "Total profit after OPEX+debt (€)"))

    rows = []
    for col, label in metrics:
        if col not in sim_summary.columns:
            continue
        rows.append({
            "metric": label,
            "mean": mean(col),
            "p10": q(col, 0.10),
            "p50": q(col, 0.50),
            "p90": q(col, 0.90),
        })

    out = pd.DataFrame(rows)

    # Add key probabilities commonly used in investment decisions
    if "npv_eur" in sim_summary.columns:
        s = pd.to_numeric(sim_summary["npv_eur"], errors="coerce").dropna()
        out = pd.concat([out, pd.DataFrame([{
            "metric": "Prob(NPV < 0)",
            "mean": float((s < 0).mean()) if not s.empty else float("nan"),
            "p10": np.nan, "p50": np.nan, "p90": np.nan,
        }])], ignore_index=True)

    if "irr" in sim_summary.columns:
        s = pd.to_numeric(sim_summary["irr"], errors="coerce").dropna()
        out = pd.concat([out, pd.DataFrame([{
            "metric": f"Prob(IRR < WACC={wacc:.4f})",
            "mean": float((s < wacc).mean()) if not s.empty else float("nan"),
            "p10": np.nan, "p50": np.nan, "p90": np.nan,
        }])], ignore_index=True)

    return out


# =========================
# Main
# =========================
def power_to_profit(
    mwh_monthly_20y,
    turbine_type_id: int = 1,   # 0=LOW_WIND, 1=BALANCED, 2=HIGH_WIND
    n_turbines: int = 12,
    hub_height_m: float = 160,
    equity_eur: float | None = None,  # absolute €
    debt_rate: float = 0.045,
    equity_return: float = 0.085,
    debt_tenor_years: int = 20,
    discount_rate: float | None = None,
    revenue_kwargs: dict | None = None,
    include_monthly_revenue: bool = False,  # big output toggle
) -> dict:
    """
    Run full MC project evaluation.

    revenue_kwargs MUST include:
      - tso_id
      - eeg_on
      - cod_date


    Returns:
      - yearly_df_by_sim: (sim, project_year) incl. revenue, opex, debt, profit
      - sim_summary_df: one row per sim incl. NPV/IRR and totals
      - stats_df: diagnostic distribution stats for key outputs
      - final_summary_df: compact mean/P10/P50/P90 "final result" table
      - deterministic summaries: capex, opex_total, finance
      - optional monthly revenue table (big)
    """
    revenue_kwargs = revenue_kwargs or {}

    # --- Validate revenue kwargs ---
    required = ["tso_id", "eeg_on", "cod_date"]
    missing = [k for k in required if k not in revenue_kwargs]
    if missing:
        raise ValueError(
            f"revenue_kwargs missing required keys: {missing}. "
        )

    tso_id = int(revenue_kwargs["tso_id"])
    eeg_on = int(revenue_kwargs["eeg_on"])
    cod_date = revenue_kwargs["cod_date"]

    cod = pd.Timestamp(cod_date)
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

    # --- Normalize production input ---
    mwh_df = _normalize_mwh_input(mwh_monthly_20y)
    mwh_df["date"] = pd.to_datetime(mwh_df["date"], errors="coerce")
    mwh_df["mwh"] = pd.to_numeric(mwh_df["mwh"], errors="coerce")
    mwh_df = mwh_df.dropna(subset=["date", "mwh"]).sort_values("date").reset_index(drop=True)

    # -----------------------
    # 1) Revenue (MC one-step)
    #    (MC CSV path is hardcoded inside revenue2.py)
    # -----------------------
    revenue_res = run_revenue_mc_one_step(
        tso_id=tso_id,
        mwh_monthly=mwh_df,
        eeg_on=eeg_on,
        cod_date=cod.strftime("%Y-%m"),
        forecast_months=FORECAST_MONTHS,
        write_outputs=False,
        write_monthly=include_monthly_revenue,
    )

    annual_by_sim = revenue_res["annual_by_sim"].copy()
    totals_by_sim = revenue_res["totals_by_sim"].copy()
    monthly_by_sim = revenue_res.get("monthly_by_sim", None)

    # -----------------------
    # EEG UI meta (strike + uplift)
    # -----------------------
    eeg_strike_mean = None
    eeg_uplift_mean = 0.0
    eeg_uplift_p50 = 0.0

    eeg_meta = revenue_res.get("eeg_meta", None)

    if int(eeg_on) == 1:
        # uplift is already in totals_by_sim (per sim)
        if "eeg_premium_total_eur" in totals_by_sim.columns:
            prem = pd.to_numeric(totals_by_sim["eeg_premium_total_eur"], errors="coerce")
            eeg_uplift_mean = float(prem.mean())
            eeg_uplift_p50 = float(prem.quantile(0.5))

        # strike comes from eeg_meta table (per sim)
        if isinstance(eeg_meta, pd.DataFrame) and "eeg_strike_eur_per_mwh" in eeg_meta.columns:
            merged = totals_by_sim.merge(
                eeg_meta[["sim", "eeg_strike_eur_per_mwh"]],
                on="sim",
                how="left",
            )
            eeg_strike_mean = float(pd.to_numeric(merged["eeg_strike_eur_per_mwh"], errors="coerce").mean())


    # Sanity checks
    for col in ["sim", "project_year", "revenue_eur"]:
        if col not in annual_by_sim.columns:
            raise ValueError(f"annual_by_sim missing '{col}'. Found: {list(annual_by_sim.columns)}")

    annual_by_sim["sim"] = pd.to_numeric(annual_by_sim["sim"], errors="coerce").astype(int)
    annual_by_sim["project_year"] = pd.to_numeric(annual_by_sim["project_year"], errors="coerce").astype(int)
    annual_by_sim["revenue_eur"] = pd.to_numeric(annual_by_sim["revenue_eur"], errors="coerce").fillna(0.0)

    sims = np.sort(annual_by_sim["sim"].unique())
    horizon_years = int(np.ceil(FORECAST_MONTHS / 12.0))
    year_starts = [cod + pd.DateOffset(months=12 * (y - 1)) for y in range(1, horizon_years + 1)]

    # -----------------------
    # 2) CAPEX (deterministic)
    # -----------------------
    capex_res = capex_mod.windpark_capex(
        n_turbines=n_turbines,
        turbine_type_id=turbine_type_id,
        hub_height_m=hub_height_m,
    )
    capex_total = float(capex_res["total_capex_eur"])
    park_mw = float(capex_res["park_mw"])

    # -----------------------
    # 3) OPEX (deterministic per project year)
    # -----------------------
    opex_df, opex_total = opex_mod.windpark_opex_timeseries(
        park_mw=park_mw,
        forecast_months=FORECAST_MONTHS,
    )
    if "project_year" not in opex_df.columns or "annual_opex_eur" not in opex_df.columns:
        raise ValueError("OPEX output must contain columns: 'project_year' and 'annual_opex_eur'.")
    opex_df = opex_df.sort_values("project_year").reset_index(drop=True)

    # -----------------------
    # 4) Finance (deterministic debt schedule)
    # -----------------------
    if equity_eur is None:
        equity_eur = 0.15 * capex_total

    fin = finance_mod.financing_model(
        capex_eur=capex_total,
        equity_eur=float(equity_eur),
        debt_rate=debt_rate,
        equity_return=equity_return,
        debt_tenor_years=debt_tenor_years,
        forecast_months=FORECAST_MONTHS,
    )

    debt_service_df = fin["debt_service_yearly_df"]
    if "project_year" not in debt_service_df.columns or "debt_service_eur" not in debt_service_df.columns:
        raise ValueError("finance.py must return debt_service_yearly_df with columns: project_year, debt_service_eur")
    debt_service_df = debt_service_df.sort_values("project_year").reset_index(drop=True)

    # -----------------------
    # 5) Build deterministic year panel
    # -----------------------
    det = pd.DataFrame({
        "project_year": list(range(1, horizon_years + 1)),
        "year_start": year_starts,
    })
    det = det.merge(opex_df[["project_year", "annual_opex_eur"]], on="project_year", how="left")
    det = det.merge(debt_service_df[["project_year", "debt_service_eur"]], on="project_year", how="left")

    det["annual_opex_eur"] = det["annual_opex_eur"].fillna(0.0)
    det["debt_service_eur"] = det["debt_service_eur"].fillna(0.0)

    # -----------------------
    # 6) Cross join sims x years, merge MC revenue
    # -----------------------
    sims_df = pd.DataFrame({"sim": sims})
    yearly = sims_df.merge(det, how="cross")

    yearly = yearly.merge(
        annual_by_sim[["sim", "project_year", "revenue_eur"]],
        on=["sim", "project_year"],
        how="left",
    )
    yearly["revenue_eur"] = yearly["revenue_eur"].fillna(0.0)

    yearly["profit_after_opex_and_debt_eur"] = (
        yearly["revenue_eur"] - yearly["annual_opex_eur"] - yearly["debt_service_eur"]
    )

    # Total profit per sim (useful for summaries)
    profit_totals = (
        yearly.groupby("sim", as_index=False)["profit_after_opex_and_debt_eur"]
             .sum()
             .rename(columns={"profit_after_opex_and_debt_eur": "total_profit_after_opex_and_debt_eur"})
    )

    # -----------------------
    # 7) Equity cashflows + NPV/IRR per sim
    # -----------------------
    used_discount_rate = float(discount_rate) if discount_rate is not None else float(fin["wacc"])
    wacc = float(fin["wacc"])
    terminal_value = float(TERMINAL_VALUE_SHARE * capex_total)

    sim_rows = []
    for sim in sims:
        y = yearly.loc[yearly["sim"] == sim].sort_values("project_year")
        equity_cashflows = [-float(equity_eur)] + y["profit_after_opex_and_debt_eur"].astype(float).tolist()
        equity_cashflows[-1] += terminal_value

        npv_eur = _compute_npv(equity_cashflows, used_discount_rate)
        irr = _safe_irr(equity_cashflows)

        sim_rows.append({
            "sim": int(sim),
            "npv_eur": float(npv_eur),
            "irr": float(irr),
            "equity_eur": float(equity_eur),
            "terminal_value_eur": float(terminal_value),
            "discount_rate_used": float(used_discount_rate),
        })

    sim_summary = pd.DataFrame(sim_rows).sort_values("sim").reset_index(drop=True)

    # Merge in revenue totals from revenue runner (if present)
    if "sim" in totals_by_sim.columns:
        sim_summary = sim_summary.merge(totals_by_sim, on="sim", how="left")

    # Merge in profit totals (computed here)
    sim_summary = sim_summary.merge(profit_totals, on="sim", how="left")

    # -----------------------
    # 8) Stats diagnostics + final summary table
    # -----------------------
    stats_frames = [
        stats_table(sim_summary["npv_eur"], "npv_eur"),
        stats_table(sim_summary["irr"], "irr", wacc=wacc),
    ]

    if "revenue_total_eur" in sim_summary.columns:
        stats_frames.append(stats_table(sim_summary["revenue_total_eur"], "revenue_total_eur"))

    if "total_profit_after_opex_and_debt_eur" in sim_summary.columns:
        stats_frames.append(stats_table(sim_summary["total_profit_after_opex_and_debt_eur"], "total_profit_after_opex_and_debt_eur"))

    stats_df = pd.concat(stats_frames, ignore_index=True)
    final_summary_df = scenario_summary(sim_summary, wacc=wacc)

    return {
        # Optional (big)
        "monthly_revenue_df_by_sim": monthly_by_sim,
        # Core annual panel
        "yearly_df_by_sim": yearly,
        # Per simulation
        "sim_summary_df": sim_summary,
        # Rich stats + compact "final result"
        "stats_df": stats_df,
        "final_summary_df": final_summary_df,
        # Deterministic summaries
        "capex_summary": {**capex_res},
        "opex_summary": {"total_opex_eur": float(opex_total)},
        "finance_summary": fin | {"equity_eur_input": float(equity_eur)},
        # Scalars
        "discount_rate_used": float(used_discount_rate),
        "terminal_value_eur": float(terminal_value),
        "n_sims": int(len(sims)),
        # EEG (for UI)
        "eeg_strike_eur_per_mwh": eeg_strike_mean,
        "eeg_uplift_total_eur_mean": float(eeg_uplift_mean),
        "eeg_uplift_total_eur_p50": float(eeg_uplift_p50),

    }


# -------------------------
# Example call
# -------------------------
if __name__ == "__main__":
    cod = pd.Timestamp("2026-01-01")
    dates = pd.date_range(cod, periods=FORECAST_MONTHS, freq="MS")
    example_mwh = pd.Series(24000.0, index=dates)

    result = power_to_profit(
        mwh_monthly_20y=example_mwh,
        turbine_type_id=1,
        n_turbines=12,
        hub_height_m=160,
        equity_eur=60_000_000,
        revenue_kwargs={
            "tso_id": 0,
            "eeg_on": 1,
            "cod_date": str(cod.date()),
        },
        include_monthly_revenue=False,
    )

    pd.set_option("display.width", 1000)

    print("\n=== FINAL SUMMARY (mean / P10 / P50 / P90) ===")
    print(result["final_summary_df"])

    print("\n=== STATS (distribution diagnostics) ===")
    print(result["stats_df"].T)  # transpose for readability in console

    print("\nSim summary head:")
    print(result["sim_summary_df"].head())
