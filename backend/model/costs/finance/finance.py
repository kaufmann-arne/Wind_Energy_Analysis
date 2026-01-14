import numpy as np

def financing_model(
    capex_eur,
    equity_share=0.15,
    debt_rate=0.045,
    equity_return=0.08,
    debt_tenor_years=20
):
    """
    Simple project finance model for a wind park.
    """

    debt_share = 1 - equity_share

    equity = capex_eur * equity_share
    debt = capex_eur * debt_share

    # annuity loan formula
    r = debt_rate
    n = debt_tenor_years
    annuity_factor = r * (1 + r)**n / ((1 + r)**n - 1)
    annual_debt_service = debt * annuity_factor

    # total debt repaid
    total_debt_repayment = annual_debt_service * n
    total_interest = total_debt_repayment - debt

    # Weighted Average Cost of Capital
    wacc = equity_share * equity_return + debt_share * debt_rate

    return {
        "capex_eur": capex_eur,
        "equity_share": equity_share,
        "debt_share": debt_share,
        "equity_eur": equity,
        "debt_eur": debt,
        "annual_debt_service_eur": annual_debt_service,
        "total_interest_paid_eur": total_interest,
        "wacc": wacc
    }

# -------------------------------
# Realistic German wind park example
# -------------------------------
if __name__ == "__main__":
    capex = 80_000_000          # 80 million € project
    equity_share = 0.15        # 15% equity (aggressive but bankable)
    debt_rate = 0.045          # 4.5% KfW-270 loan
    equity_return = 0.085      # 8.5% equity target
    tenor = 18                 # 18-year loan

    result = financing_model(
        capex_eur=capex,
        equity_share=equity_share,
        debt_rate=debt_rate,
        equity_return=equity_return,
        debt_tenor_years=tenor
    )

    print("---- German Onshore Wind Financing ----")
    print(f"CAPEX:            {result['capex_eur']/1e6:.1f} M€")
    print(f"Equity (15%):     {result['equity_eur']/1e6:.1f} M€")
    print(f"Debt (85%):       {result['debt_eur']/1e6:.1f} M€")
    print(f"Annual debt svc: {result['annual_debt_service_eur']/1e6:.2f} M€ / year")
    print(f"Total interest:  {result['total_interest_paid_eur']/1e6:.1f} M€ over {tenor} years")
    print(f"WACC:            {result['wacc']*100:.2f} %")
