import pandas as pd

OPEX_DECADE_1 = 53   # €/kW/year (years 1–10)
OPEX_DECADE_2 = 55   # €/kW/year (years 11–20)

def windpark_opex_timeseries(park_mw, years=20):
    """
    Returns a DataFrame with yearly O&M costs and total lifetime OPEX.
    """
    park_kw = park_mw * 1000
    data = []

    for year in range(1, years + 1):
        if year <= 10:
            opex_per_kw = OPEX_DECADE_1
        else:
            opex_per_kw = OPEX_DECADE_2

        annual_cost = park_kw * opex_per_kw

        data.append({
            "year": year,
            "opex_eur_per_kw": opex_per_kw,
            "annual_opex_eur": annual_cost
        })

    df = pd.DataFrame(data)
    total = df["annual_opex_eur"].sum()

    return df, total

df, total = windpark_opex_timeseries(park_mw=50)

print(df)
print("Total OPEX over 20 years:", total/1e6, "million €")
