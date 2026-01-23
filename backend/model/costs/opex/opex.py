# opex.py
#
# OPEX calculator for wind park projects.
# Produces monthly OPEX and aggregates to project-year blocks (12-month periods).

from __future__ import annotations

import pandas as pd

# Annual OPEX rates in €/kW/year (real terms)
OPEX_DECADE_1 = 53.0  # project years 1–10
OPEX_DECADE_2 = 55.0  # project years 11+ (flat beyond year 10 unless you change logic)

MONTHS_PER_YEAR = 12


def windpark_opex_timeseries(
    *,
    park_mw: float,
    forecast_months: int = 240,
    opex_decade_1: float = OPEX_DECADE_1,
    opex_decade_2: float = OPEX_DECADE_2,
) -> tuple[pd.DataFrame, float]:
    """
    Compute OPEX time series aligned to project months and aggregated by project-year.

    Project year definition:
        months 1–12 -> year 1, months 13–24 -> year 2, etc.

    Parameters
    ----------
    park_mw:
        Park capacity in MW. Must be > 0.
    forecast_months:
        Number of months in horizon (default 240 = 20 years). Must be > 0.
    opex_decade_1:
        €/kW/year applied for project years 1–10.
    opex_decade_2:
        €/kW/year applied for project years 11+.

    Returns
    -------
    yearly_df, total_opex_eur
        yearly_df columns:
        - project_year
        - annual_opex_eur
        - opex_eur_per_kw_year
        - months_in_year
    """
    park_mw = float(park_mw)
    if park_mw <= 0:
        raise ValueError("park_mw must be > 0")

    forecast_months = int(forecast_months)
    if forecast_months <= 0:
        raise ValueError("forecast_months must be > 0")

    park_kw = park_mw * 1000.0

    # Monthly index: 1..N
    m = pd.DataFrame({"project_month": range(1, forecast_months + 1)})
    m["project_year"] = ((m["project_month"] - 1) // MONTHS_PER_YEAR) + 1

    # Vectorized decade rate selection (no apply/lambda)
    m["opex_eur_per_kw_year"] = (
        (m["project_year"] <= 10)
        .map({True: float(opex_decade_1), False: float(opex_decade_2)})
        .astype(float)
    )

    # Annual €/kW/year -> monthly EUR
    m["monthly_opex_eur"] = park_kw * m["opex_eur_per_kw_year"] / MONTHS_PER_YEAR

    total_opex_eur = float(m["monthly_opex_eur"].sum())

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


if __name__ == "__main__":
    yearly, total = windpark_opex_timeseries(park_mw=50, forecast_months=240)
    print(yearly.head(12))
    print("Total OPEX:", total / 1e6, "million €")
