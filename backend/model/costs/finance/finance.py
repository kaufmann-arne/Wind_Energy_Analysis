"""
finance.py

Simple project finance model for a wind park.

Inputs typically come from profit.py:
- capex_eur (from capex_res["total_capex_eur"])
- equity_eur (absolute € amount) OR equity_share (0..1)
- debt_rate, equity_return, debt_tenor_years
- forecast_months (project horizon; default 240)

Outputs:
- Derived equity share and debt share
- Annual annuity-style debt service
- Debt service schedule by project year aligned to the project horizon
- Simple WACC

Notes / assumptions:
- Debt service is modeled as a fully amortizing annuity loan (principal + interest),
  with constant annual payments during the tenor.
- After the debt tenor ends, debt service is 0.
- WACC is a simple weighted blend using debt_rate and equity_return
  (no tax shield, fees, DSRA, sculpting, refinancing, etc.).
"""

from __future__ import annotations

import math
from typing import Final

import pandas as pd


MONTHS_PER_YEAR: Final[int] = 12
DEFAULT_EQUITY_SHARE: Final[float] = 0.15


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp numeric value to [lo, hi]."""
    return max(lo, min(value, hi))


def financing_model(
    *,
    capex_eur: float,
    equity_eur: float | None = None,
    # If equity_eur is None, you can still pass equity_share directly
    equity_share: float | None = None,
    debt_rate: float = 0.045,
    equity_return: float = 0.08,
    debt_tenor_years: int = 20,
    forecast_months: int = 240,
) -> dict:
    """
    Compute a basic financing structure and debt service schedule.

    Parameters
    ----------
    capex_eur:
        Total project CAPEX in EUR (> 0).
    equity_eur:
        Absolute equity amount in EUR (>= 0). If provided, equity_share is derived as equity_eur/capex_eur.
        If equity_eur > capex_eur, equity_share is clamped to 1.0 (100% equity, 0% debt).
    equity_share:
        Optional direct equity share in [0..1]. Used only if equity_eur is None.
        If both equity_eur and equity_share are provided, equity_eur takes precedence.
    debt_rate:
        Annual debt interest rate (>= 0), e.g. 0.045.
    equity_return:
        Annual equity target return (used for WACC).
    debt_tenor_years:
        Loan tenor in years (> 0). Annual annuity debt service is assumed.
    forecast_months:
        Project horizon in months (> 0). Used to align the debt schedule to the same project-year blocks.

    Returns
    -------
    dict with keys used by power_to_profit.py plus schedule:
      - annual_debt_service_eur
      - wacc
      - equity_share_derived
      - debt_service_yearly_df (project_year, debt_service_eur)
      - plus breakdown fields (equity_eur, debt_eur, total_interest_paid_eur, etc.)
    """
    # ---- Validate core inputs ----
    capex_eur = float(capex_eur)
    if capex_eur <= 0:
        raise ValueError("capex_eur must be > 0")

    forecast_months = int(forecast_months)
    if forecast_months <= 0:
        raise ValueError("forecast_months must be > 0")

    debt_rate = float(debt_rate)
    if debt_rate < 0:
        raise ValueError("debt_rate must be >= 0")

    equity_return = float(equity_return)

    debt_tenor_years = int(debt_tenor_years)
    if debt_tenor_years <= 0:
        raise ValueError("debt_tenor_years must be > 0")

    # ---- Determine equity share (equity_eur overrides equity_share) ----
    if equity_eur is not None:
        equity_eur = float(equity_eur)
        if equity_eur < 0:
            raise ValueError("equity_eur must be >= 0")

        # Derive share; if equity exceeds CAPEX, we'll clamp to 100% equity below.
        equity_share_derived = equity_eur / capex_eur
    else:
        if equity_share is None:
            equity_share_derived = DEFAULT_EQUITY_SHARE
        else:
            equity_share_derived = float(equity_share)

    # ---- What if equity > 100%? ----
    # We clamp the equity share to 1.0, which means:
    # - debt_share becomes 0
    # - debt_eur becomes 0
    # - annual debt service becomes 0
    #
    # The *excess* equity above CAPEX is not modeled as "extra cash" here; we simply
    # cap the financed amount at CAPEX. (If you want to track excess equity as cash,
    # we can add an "excess_equity_eur" output.)
    equity_share_derived = _clamp(equity_share_derived, 0.0, 1.0)

    debt_share = 1.0 - equity_share_derived
    equity_eur_final = capex_eur * equity_share_derived
    debt_eur = capex_eur * debt_share

    # ---- Compute annual annuity debt service ----
    r = debt_rate
    n = debt_tenor_years

    if debt_eur <= 0:
        annual_debt_service_eur = 0.0
        total_interest_paid_eur = 0.0
    else:
        if r == 0.0:
            # Zero-interest: straight-line principal repayment
            annual_debt_service_eur = debt_eur / n
            total_interest_paid_eur = 0.0
        else:
            # Standard annuity payment formula
            # Payment = Principal * [ r(1+r)^n / ((1+r)^n - 1) ]
            annuity_factor = r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)
            annual_debt_service_eur = debt_eur * annuity_factor

            total_debt_repayment = annual_debt_service_eur * n
            total_interest_paid_eur = total_debt_repayment - debt_eur

    # ---- WACC (simple weighted blend) ----
    wacc = equity_share_derived * equity_return + debt_share * debt_rate

    # ---- Build aligned debt schedule by project year ----
    horizon_years = int(math.ceil(forecast_months / MONTHS_PER_YEAR))
    years = pd.Series(range(1, horizon_years + 1), name="project_year")

    # Vectorized schedule: pay during tenor, then 0
    debt_service = (years <= debt_tenor_years).astype(float) * float(annual_debt_service_eur)

    debt_service_yearly_df = pd.DataFrame(
        {"project_year": years, "debt_service_eur": debt_service}
    )

    return {
        "capex_eur": float(capex_eur),
        "equity_share_derived": float(equity_share_derived),
        "debt_share": float(debt_share),
        "equity_eur": float(equity_eur_final),
        "debt_eur": float(debt_eur),
        "debt_rate": float(debt_rate),
        "equity_return": float(equity_return),
        "debt_tenor_years": int(debt_tenor_years),
        "annual_debt_service_eur": float(annual_debt_service_eur),
        "total_interest_paid_eur": float(total_interest_paid_eur),
        "wacc": float(wacc),
        "forecast_months": int(forecast_months),
        "horizon_years": int(horizon_years),
        "debt_service_yearly_df": debt_service_yearly_df,
    }


if __name__ == "__main__":
    capex = 80_000_000
    equity_eur = 90_000_000  # intentionally > CAPEX to demonstrate clamping
    debt_rate = 0.045
    equity_return = 0.085
    tenor = 20
    forecast_months = 240

    result = financing_model(
        capex_eur=capex,
        equity_eur=equity_eur,
        debt_rate=debt_rate,
        equity_return=equity_return,
        debt_tenor_years=tenor,
        forecast_months=forecast_months,
    )

    print("---- Financing ----")
    print(f"CAPEX:               {result['capex_eur']/1e6:.1f} M€")
    print(f"Equity (input):      {equity_eur/1e6:.1f} M€")
    print(f"Equity used:         {result['equity_eur']/1e6:.1f} M€ (clamped to CAPEX)")
    print(f"Equity share:        {result['equity_share_derived']*100:.1f} %")
    print(f"Debt:                {result['debt_eur']/1e6:.1f} M€")
    print(f"Annual debt service: {result['annual_debt_service_eur']/1e6:.2f} M€ / year")
    print(f"Total interest:      {result['total_interest_paid_eur']/1e6:.1f} M€ over {tenor} years")
    print(f"WACC:                {result['wacc']*100:.2f} %")
    print(result["debt_service_yearly_df"].head())
