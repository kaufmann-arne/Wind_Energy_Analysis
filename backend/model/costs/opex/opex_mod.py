# opex.py
#
# Central callable OPEX module for power_to_profit.py
# - Inputs come from profit.py:
#     - park_mw (already computed from capex_res["park_mw"])
#     - forecast_months (typically 240 for 20 years)
# - Produces monthly + project-year OPEX aligned to the same 12-month blocks as revenue.

import pandas as pd

# €/kW/year (real terms)
OPEX_DECADE_1 = 53.0   # years 1–10
OPEX_DECADE_2 = 55.0   # years 11–20


def windpark_opex_timeseries(
    *,
    park_mw: float,
    forecast_months: int = 240,
) -> tuple[pd.DataFrame, float]:
    """
    Compute OPEX time series aligned to project months/years.

    Parameters
    ----------
    park_mw : float
        Park capacity in MW.
    forecast_months : int
        Number of months in horizon (default 240 = 20y).

    Returns
    -------
    yearly_df : DataFrame with columns:
        - project_year (1..ceil(forecast_months/12))
        - annual_opex_eur
        - opex_eur_per_kw_year  (rate applied for that project year)
    total_opex_eur : float
        Total OPEX over the horizon (sum of monthly, equivalently yearly).
    """
    park_mw = float(park_mw)
    if park_mw <= 0:
        raise ValueError("park_mw must be > 0")

    forecast_months = int(forecast_months)
    if forecast_months <= 0:
        raise ValueError("forecast_months must be > 0")

    park_kw = park_mw * 1000.0

    # Build monthly series (project month 1..N)
    m = pd.DataFrame({"project_month": range(1, forecast_months + 1)})
    m["project_year"] = ((m["project_month"] - 1) // 12) + 1  # 1..20 (for 240)

    # Choose rate per project year (decade 1 vs decade 2)
    m["opex_eur_per_kw_year"] = m["project_year"].apply(
        lambda y: OPEX_DECADE_1 if y <= 10 else OPEX_DECADE_2
    ).astype(float)

    # Convert annual €/kW/year into monthly €:
    # monthly_cost = park_kw * (€/kW/year) / 12
    m["monthly_opex_eur"] = park_kw * m["opex_eur_per_kw_year"] / 12.0

    total_opex_eur = float(m["monthly_opex_eur"].sum())

    # Aggregate to project-year (sum of months in that year; last year might be partial if forecast_months not multiple of 12)
    yearly_df = (
        m.groupby("project_year", as_index=False)
         .agg(
            annual_opex_eur=("monthly_opex_eur", "sum"),
            opex_eur_per_kw_year=("opex_eur_per_kw_year", "first"),
            months_in_year=("monthly_opex_eur", "size"),
         )
         .sort_values("project_year")
         .reset_index(drop=True)
    )

    return yearly_df, total_opex_eur


# -----------------------
# Optional standalone test
# -----------------------
if __name__ == "__main__":
    yearly, total = windpark_opex_timeseries(park_mw=50, forecast_months=240)
    print(yearly.head(12))
    print("Total OPEX:", total / 1e6, "million €")
