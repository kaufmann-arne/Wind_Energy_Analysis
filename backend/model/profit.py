# power_to_profit.py

import numpy as np
import pandas as pd
import numpy_financial as npf
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Add module folders to Python import path
sys.path.insert(0, str(ROOT / "revenue"))
sys.path.insert(0, str(ROOT / "costs" / "capex"))
sys.path.insert(0, str(ROOT / "costs" / "opex"))
sys.path.insert(0, str(ROOT / "costs" / "finance"))

import revenue_mod as revenue_mod
import capex_mod as capex_mod
import opex_mod as opex_mod
import finance_mod as finance_mod


FORECAST_MONTHS = 240  # 20 years


def _compute_npv(cashflows, discount_rate):
    return sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cashflows))



def _compute_irr(cashflows, guess=0.08):
    if all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
        return np.nan

    r = guess
    for _ in range(200):
        npv = 0.0
        d_npv = 0.0
        for t, cf in enumerate(cashflows):
            denom = (1 + r) ** t
            npv += cf / denom
            if t > 0:
                d_npv += -t * cf / ((1 + r) ** (t + 1))

        if abs(npv) < 1e-6:
            return r

        if abs(d_npv) < 1e-12:
            break

        r_new = r - (npv / d_npv)
        # If Newton wants to run outside sane bounds, treat as failure
        if r_new <= -0.99 or r_new >= 5.0:
            return np.nan


        if abs(r_new - r) < 1e-9:
            return r_new

        r = r_new

    return np.nan


def _month_index(dt: pd.Series) -> pd.Series:
    """Integer month index for easy month-diff calculations: year*12 + (month-1)."""
    return dt.dt.year * 12 + (dt.dt.month - 1)


def power_to_profit(
    mwh_monthly_20y,
    turbine_type_id=1,   # 0=LOW_WIND, 1=BALANCED, 2=HIGH_WIND
    n_turbines=12,
    hub_height_m=160,
    equity_eur=None,     # <-- absolute € amount (not share)
    debt_rate=0.045,
    equity_return=0.085,
    debt_tenor_years=18,
    discount_rate=None,
    revenue_kwargs=None,
):
    """
    Orchestrates: revenue + capex + opex + finance => yearly profit, NPV, IRR

    IMPORTANT:
    - Yearly aggregation is PROJECT YEARS (12-month blocks from COD),
      not calendar years.
    """

    revenue_kwargs = revenue_kwargs or {}

    # -----------------------
    # 1) Prepare monthly MWh input -> DataFrame(date, mwh)
    # -----------------------
    if isinstance(mwh_monthly_20y, pd.Series):
        mwh_df = mwh_monthly_20y.rename("mwh").to_frame()
        mwh_df = mwh_df.reset_index().rename(columns={"index": "date"})
    elif isinstance(mwh_monthly_20y, pd.DataFrame):
        mwh_df = mwh_monthly_20y.copy()
        if "date" not in mwh_df.columns:
            raise ValueError("mwh_monthly_20y DataFrame must contain a 'date' column.")
        if "mwh" not in mwh_df.columns:
            for alt in ["mwh_gross", "MWh", "MWh_gross", "mwh_gross_park", "mwh_park"]:
                if alt in mwh_df.columns:
                    mwh_df = mwh_df.rename(columns={alt: "mwh"})
                    break
        if "mwh" not in mwh_df.columns:
            raise ValueError("mwh_monthly_20y must contain an 'mwh' column (or mwh_gross).")
    else:
        raise TypeError("mwh_monthly_20y must be a pandas Series or DataFrame.")

    mwh_df["date"] = pd.to_datetime(mwh_df["date"], errors="coerce")
    mwh_df["mwh"] = pd.to_numeric(mwh_df["mwh"], errors="coerce")
    mwh_df = mwh_df.dropna(subset=["date", "mwh"]).sort_values("date").reset_index(drop=True)

    # -----------------------
    # 2) Revenue module (NEW API)
    # -----------------------
    required = ["tso_id", "eeg_on", "cod_date"]
    missing = [k for k in required if k not in revenue_kwargs]
    if missing:
        raise ValueError(
            f"revenue_kwargs missing required keys: {missing}. "
            f"Expected at least: {required} (manual_eeg_strike optional)."
        )

    tso_id = int(revenue_kwargs["tso_id"])
    eeg_on = int(revenue_kwargs.get("eeg_on", 1))
    manual_eeg_strike = revenue_kwargs.get("manual_eeg_strike", None)
    cod_date = revenue_kwargs["cod_date"]

    # COD normalized to month-start
    cod = pd.Timestamp(cod_date)
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

    optional_keys = ["base_dir", "market_file", "cf_file", "curtail_file", "eeg_file"]
    optional_args = {k: revenue_kwargs[k] for k in optional_keys if k in revenue_kwargs}

    monthly_rev = revenue_mod.generate_monthly_revenue(
        tso_id=tso_id,
        mwh_monthly_20y=mwh_df,
        eeg_on=eeg_on,
        manual_eeg_strike=manual_eeg_strike,
        cod_date=cod,
        forecast_months=FORECAST_MONTHS,
        **optional_args,
    )

    if "date" not in monthly_rev.columns or "revenue_eur" not in monthly_rev.columns:
        raise ValueError("Revenue output must contain columns: 'date' and 'revenue_eur'.")

    monthly_rev["date"] = pd.to_datetime(monthly_rev["date"], errors="coerce")
    monthly_rev = monthly_rev.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print("\n===== REVENUE DRIVERS =====")
    print("Avg delivered MWh / month:", monthly_rev["mwh_delivered"].mean())
    print("Avg realised price €/MWh:", monthly_rev["p_wind_realised"].mean())
    print("Avg revenue per month (k€):", monthly_rev["revenue_eur"].mean() / 1e3)
    print("Avg market price €/MWh:", monthly_rev["p_market"].mean())
    print("Avg capture factor:", monthly_rev["cf"].mean())
    print("Avg curtailment:", monthly_rev["cr"].mean())
    print("==========================\n")


    # safety net
    monthly_rev = monthly_rev[monthly_rev["date"] >= cod].copy()
    if len(monthly_rev) < FORECAST_MONTHS:
        raise ValueError(f"Revenue series has only {len(monthly_rev)} months from COD, need {FORECAST_MONTHS}.")
    monthly_rev = monthly_rev.head(FORECAST_MONTHS).copy()

    # -----------------------
    # 2c) Aggregate by PROJECT YEAR (Year 1..20 from COD)
    # -----------------------
    moff = _month_index(monthly_rev["date"]) - (cod.year * 12 + (cod.month - 1))
    monthly_rev["project_month"] = moff + 1
    monthly_rev["project_year"] = (moff // 12) + 1

    yearly_revenue = (
        monthly_rev.groupby("project_year", as_index=False)["revenue_eur"]
        .sum()
        .sort_values("project_year")
        .reset_index(drop=True)
    )

    horizon_years = int(np.ceil(FORECAST_MONTHS / 12.0))
    year_starts = [cod + pd.DateOffset(months=12 * (y - 1)) for y in range(1, horizon_years + 1)]
    yearly_revenue["year_start"] = year_starts

    # -----------------------
    # 3) CAPEX module  (CAPEX -> finance dependency handled correctly)
    # -----------------------
    capex_res = capex_mod.windpark_capex(
        n_turbines=n_turbines,
        turbine_type_id=turbine_type_id,
        hub_height_m=hub_height_m,
    )
    capex_total = float(capex_res["total_capex_eur"])
    park_mw = float(capex_res["park_mw"])

    # -----------------------
    # 4) OPEX module (NEW: aligned to forecast_months, returns project_year already)
    # -----------------------
    opex_df, opex_total = opex_mod.windpark_opex_timeseries(
        park_mw=park_mw,
        forecast_months=FORECAST_MONTHS,
    )

    # NEW opex.py returns: project_year, annual_opex_eur
    if "project_year" not in opex_df.columns or "annual_opex_eur" not in opex_df.columns:
        raise ValueError("OPEX output must contain columns: 'project_year' and 'annual_opex_eur'.")

    opex_df = opex_df.sort_values("project_year").reset_index(drop=True)

    # -----------------------
    # 5) Financing module (NEW: equity_eur input, debt schedule returned)
    # -----------------------
    if equity_eur is None:
        equity_eur = 0.15 * capex_total  # default: 15% of capex

    fin = finance_mod.financing_model(
        capex_eur=capex_total,
        equity_eur=float(equity_eur),         # <-- absolute €
        debt_rate=debt_rate,
        equity_return=equity_return,
        debt_tenor_years=debt_tenor_years,
        forecast_months=FORECAST_MONTHS,
    )

    # aligned yearly debt schedule
    debt_service_df = fin["debt_service_yearly_df"]
    if "project_year" not in debt_service_df.columns or "debt_service_eur" not in debt_service_df.columns:
        raise ValueError("finance.py must return debt_service_yearly_df with columns: project_year, debt_service_eur")

    debt_service_df = debt_service_df.sort_values("project_year").reset_index(drop=True)

    print("\n===== DEBT SERVICE SCHEDULE =====")
    print(debt_service_df.head(5))
    print("...")
    print(debt_service_df.tail(5))
    print("================================\n")


    print("\n===== FINANCIAL SANITY CHECK =====")
    print("Park MW:", park_mw)
    print("CAPEX total (M€):", capex_total / 1e6)
    print("CAPEX €/kW:", capex_res["total_capex_eur_per_kw"])
    print("Equity (M€):", equity_eur / 1e6)
    print("Equity share:", fin.get("equity_share_derived"))
    print("Debt (M€):", fin["debt_eur"] / 1e6)

    print("Average yearly debt service (M€):",
        debt_service_df["debt_service_eur"].mean() / 1e6)

    print("=================================\n")


    # -----------------------
    # 6) Unite yearly cashflows (by project_year)
    # -----------------------
    yearly = pd.DataFrame({
        "project_year": list(range(1, horizon_years + 1)),
        "year_start": year_starts,
    })

    yearly = yearly.merge(yearly_revenue[["project_year", "revenue_eur"]], on="project_year", how="left")
    yearly = yearly.merge(opex_df[["project_year", "annual_opex_eur"]], on="project_year", how="left")
    yearly = yearly.merge(debt_service_df[["project_year", "debt_service_eur"]], on="project_year", how="left")

    # Fill any missing (shouldn't happen, but keeps things robust)
    yearly["revenue_eur"] = yearly["revenue_eur"].fillna(0.0)
    yearly["annual_opex_eur"] = yearly["annual_opex_eur"].fillna(0.0)
    yearly["debt_service_eur"] = yearly["debt_service_eur"].fillna(0.0)

    yearly["profit_after_opex_and_debt_eur"] = (
        yearly["revenue_eur"] - yearly["annual_opex_eur"] - yearly["debt_service_eur"]
    )

    # -----------------------
    # 7) Equity cashflows + NPV + IRR
    # -----------------------
    equity_cashflows = [-float(equity_eur)] + yearly["profit_after_opex_and_debt_eur"].tolist()

    # -----------------------
    # Terminal value (residual value at end of year 20)
    # -----------------------
    TERMINAL_VALUE_SHARE = 0.30   # 10% of CAPEX as salvage / repowering option
    terminal_value = TERMINAL_VALUE_SHARE * capex_total

    equity_cashflows[-1] += terminal_value

    print(f"Terminal value added in year 20: {terminal_value/1e6:.1f} M€")


    used_discount_rate = discount_rate if discount_rate is not None else float(fin["wacc"])
    npv_eur = _compute_npv(equity_cashflows, used_discount_rate)
    irr = npf.irr(equity_cashflows)

    return {
        "monthly_revenue_df": monthly_rev,
        "yearly_df": yearly,
        "capex_summary": {**capex_res},
        "opex_summary": {"total_opex_eur": float(opex_total)},
        "finance_summary": fin | {
            "equity_eur_input": float(equity_eur),
        },
        "equity_cashflows": equity_cashflows,
        "npv_eur": npv_eur,
        "irr": irr,
        "discount_rate_used": used_discount_rate,
    }


# -------------------------
# Example call
# -------------------------
if __name__ == "__main__":
    cod = pd.Timestamp.today()
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

    dates = pd.date_range(cod, periods=FORECAST_MONTHS, freq="MS")
    example_mwh = pd.Series(24000.0, index=dates)

    result = power_to_profit(
        mwh_monthly_20y=example_mwh,
        turbine_type_id=1,
        n_turbines=12,
        hub_height_m=160,
        equity_eur=60_000_000,
        revenue_kwargs={
            "tso_id": 1,
            "eeg_on": 1,
            "manual_eeg_strike": None,
            "cod_date": str(cod.date()),
            # If you run from a different working dir, set:
            # "base_dir": str(ROOT),
        }
    )

    print("NPV (equity):", f"{result['npv_eur']/1e6:.1f} M€")
    print("IRR (equity):", f"{result['irr']*100:.2f}%")
    print(result["yearly_df"].head())
    print(result["yearly_df"].tail(5))


