# finance.py
#
# Central callable finance module for power_to_profit.py
# - Inputs come from profit.py:
#     - capex_eur (from capex_res["total_capex_eur"])
#     - equity_eur (absolute € amount; we compute equity_share)
#     - debt_rate, equity_return, debt_tenor_years
#     - forecast_months (project horizon; default 240)
# - Outputs include:
#     - equity_share_derived
#     - annual_debt_service_eur
#     - debt_service_yearly_df (project_year schedule aligned to horizon)
#     - wacc
#
# Notes:
# - We keep the debt service as an ANNUITY yearly payment (like your current model).
# - Horizon alignment: we build project_year 1..ceil(forecast_months/12) and set payments
#   to 0 after debt_tenor_years.

import math
import pandas as pd


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
    Simple project finance model for a wind park.

    Parameters
    ----------
    capex_eur : float
        Total project CAPEX in €.
    equity_eur : float | None
        Absolute equity amount in €. If provided, equity_share is derived as equity_eur/capex_eur.
    equity_share : float | None
        Optional direct equity share (0..1). Used only if equity_eur is None.
    debt_rate : float
        Annual debt interest rate (e.g., 0.045).
    equity_return : float
        Annual equity target return (used for WACC).
    debt_tenor_years : int
        Loan tenor in years (annuity payments).
    forecast_months : int
        Project horizon in months (default 240 = 20y). Used to create aligned debt schedule.

    Returns
    -------
    dict with keys used by power_to_profit.py plus schedule:
      - annual_debt_service_eur
      - wacc
      - equity_share_derived
      - debt_service_yearly_df (project_year, debt_service_eur)
      - plus breakdown fields (equity_eur, debt_eur, total_interest_paid_eur, etc.)
    """

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

    # -----------------------
    # Derive equity share
    # -----------------------
    if equity_eur is not None:
        equity_eur = float(equity_eur)
        if equity_eur < 0:
            raise ValueError("equity_eur must be >= 0")
        equity_share_derived = equity_eur / capex_eur
    else:
        if equity_share is None:
            # default: 15% equity if nothing provided
            equity_share_derived = 0.15
        else:
            equity_share_derived = float(equity_share)

    # clamp to [0, 1)
    equity_share_derived = max(0.0, min(equity_share_derived, 0.999999))

    debt_share = 1.0 - equity_share_derived

    equity_eur_final = capex_eur * equity_share_derived
    debt_eur = capex_eur * debt_share

    # -----------------------
    # Annuity debt service (annual)
    # -----------------------
    r = debt_rate
    n = debt_tenor_years

    if debt_eur <= 0:
        annual_debt_service_eur = 0.0
        total_interest_paid_eur = 0.0
    else:
        if r == 0.0:
            annual_debt_service_eur = debt_eur / n
        else:
            annuity_factor = r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)
            annual_debt_service_eur = debt_eur * annuity_factor

        total_debt_repayment = annual_debt_service_eur * n
        total_interest_paid_eur = total_debt_repayment - debt_eur

    # -----------------------
    # WACC (simple weighted blend)
    # -----------------------
    wacc = equity_share_derived * equity_return + debt_share * debt_rate

    # -----------------------
    # Build aligned debt schedule by project year
    # -----------------------
    horizon_years = int(math.ceil(forecast_months / 12.0))
    years = list(range(1, horizon_years + 1))

    debt_service = [
        float(annual_debt_service_eur) if y <= debt_tenor_years else 0.0
        for y in years
    ]

    debt_service_yearly_df = pd.DataFrame({
        "project_year": years,
        "debt_service_eur": debt_service,
    })

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


# -------------------------------
# Standalone test / example
# -------------------------------
if __name__ == "__main__":
    capex = 80_000_000
    equity_eur = 12_000_000
    debt_rate = 0.045
    equity_return = 0.085
    tenor = 18
    forecast_months = 240

    result = financing_model(
        capex_eur=capex,
        equity_eur=equity_eur,
        debt_rate=debt_rate,
        equity_return=equity_return,
        debt_tenor_years=tenor,
        forecast_months=forecast_months,
    )

    print("---- German Onshore Wind Financing ----")
    print(f"CAPEX:               {result['capex_eur']/1e6:.1f} M€")
    print(f"Equity (input):      {equity_eur/1e6:.1f} M€")
    print(f"Equity share:        {result['equity_share_derived']*100:.1f} %")
    print(f"Debt:                {result['debt_eur']/1e6:.1f} M€")
    print(f"Annual debt service: {result['annual_debt_service_eur']/1e6:.2f} M€ / year")
    print(f"Total interest:      {result['total_interest_paid_eur']/1e6:.1f} M€ over {tenor} years")
    print(f"WACC:                {result['wacc']*100:.2f} %")
    print(result["debt_service_yearly_df"].head())
