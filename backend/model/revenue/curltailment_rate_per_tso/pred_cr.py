import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =============================
# CONFIG
# =============================
WIND_DAILY_FILE = "wind_onshore_daily_by_tso.csv"
LOAD_MONTHLY_FILE = "../monthly_gridload_per_tso/tso_grid_load_monthly_wide.csv"
CURTAIL_FILE = "curltailment_rate_quarterly_by_tso.csv"

FUTURE_START = "2021Q1"
FUTURE_END   = "2046Q4"

CR_FLOOR = 0.0
CR_CAP = 0.30

TSOS = ["TenneT", "50Hertz", "Amprion", "TransnetBW"]
LOAD_COL_MAP = {
    "TenneT": "load_mwh_tennet",
    "50Hertz": "load_mwh_50hertz",
    "Amprion": "load_mwh_amprion",
    "TransnetBW": "load_mwh_transnetbw",
}

# Option B growth assumptions (scenario knobs)
G_WIND = {tso: 0.01  for tso in TSOS}   # +1%/yr wind proxy
G_LOAD = {tso: 0.005 for tso in TSOS}   # +0.5%/yr load proxy

LEVEL_QTRS = 8  # last 8 quarters average = baseline level

OUTPUT_FILE = "curtailment_forecast_quarterly_by_tso_2021_2046.csv"

# =============================
# HELPERS
# =============================
def to_quarter_period_from_dates(s):
    return pd.to_datetime(s).dt.to_period("Q")

def wide_to_quarter_long(df, date_col, value_cols, value_name):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["quarter"] = to_quarter_period_from_dates(df[date_col])
    long = df.melt(id_vars="quarter", value_vars=value_cols,
                   var_name="TSO", value_name=value_name)
    return long.groupby(["quarter", "TSO"], as_index=False)[value_name].sum()

def make_season_factors(df_q, value_col):
    """
    Returns season_factor per (TSO, q_num) normalized to mean=1 within each TSO.
    """
    tmp = df_q.copy()
    tmp["q_num"] = tmp["quarter"].dt.quarter.astype(int)
    seas = tmp.groupby(["TSO", "q_num"])[value_col].mean().reset_index()
    seas["mean_tso"] = seas.groupby("TSO")[value_col].transform("mean")
    seas["season_factor"] = seas[value_col] / seas["mean_tso"]
    return seas[["TSO", "q_num", "season_factor"]]

def make_future_series(df_q, value_col, g_map, future_q, last_q):
    """
    level = mean of last LEVEL_QTRS for each TSO
    season = avg by q_num per TSO (normalized)
    growth = (1+g)^(years_ahead) where years_ahead = quarters_ahead/4
    """
    df_q = df_q.copy().sort_values(["TSO", "quarter"])

    # level per TSO (last LEVEL_QTRS) — warning-free
    levels = (
        df_q.groupby("TSO", as_index=True)[value_col]
            .apply(lambda s: float(s.tail(LEVEL_QTRS).mean()))
            .to_dict()
    )

    seas = make_season_factors(df_q, value_col)

    future = pd.MultiIndex.from_product([future_q, TSOS], names=["quarter", "TSO"]).to_frame(index=False)
    future["q_num"] = future["quarter"].dt.quarter.astype(int)

    # FIX: last_q is a scalar Period
    quarters_ahead = future["quarter"].apply(lambda p: p.ordinal) - last_q.ordinal
    years_ahead = quarters_ahead / 4.0

    future = future.merge(seas, on=["TSO", "q_num"], how="left")
    future["season_factor"] = future["season_factor"].fillna(1.0)

    out = []
    for tso in TSOS:
        g = float(g_map[tso])
        lvl = float(levels[tso])

        idx = future["TSO"] == tso
        growth = (1.0 + g) ** years_ahead[idx].values
        vals = lvl * future.loc[idx, "season_factor"].values * growth

        out.append(pd.DataFrame({
            "quarter": future.loc[idx, "quarter"].values,
            "TSO": tso,
            value_col: vals
        }))

    out = pd.concat(out, ignore_index=True).sort_values(["quarter", "TSO"]).reset_index(drop=True)
    return out

# =============================
# LOAD & PREPROCESS
# =============================
# Wind (daily -> quarterly)
wind = pd.read_csv(WIND_DAILY_FILE)
wind_q = wide_to_quarter_long(wind, "date", TSOS, "wind_mwh")

# Load (monthly -> quarterly)
load = pd.read_csv(LOAD_MONTHLY_FILE)
load = load.rename(columns={v: k for k, v in LOAD_COL_MAP.items()})
load_q = wide_to_quarter_long(load, "date", TSOS, "load_mwh")

# Curtailment (quarterly by TSO in wide columns)
curt = pd.read_csv(CURTAIL_FILE)

# Parse quarter like "2015Q1" safely
curt["quarter"] = pd.PeriodIndex(curt["quarter"].astype(str), freq="Q")

records = []
for tso in TSOS:
    records.append(pd.DataFrame({
        "quarter": curt["quarter"],
        "TSO": tso,
        "cr": pd.to_numeric(curt[f"{tso}_curtailment_rate"], errors="coerce"),
        "prod_mwh": pd.to_numeric(curt[f"{tso}_production_mwh"], errors="coerce"),
        "curt_mwh": pd.to_numeric(curt[f"{tso}_curtailed_mwh"], errors="coerce"),
    }))
curt_q = pd.concat(records, ignore_index=True).dropna(subset=["quarter", "TSO", "cr"])

# Merge historical modeling frame
df = curt_q.merge(wind_q, on=["quarter", "TSO"], how="inner").merge(load_q, on=["quarter", "TSO"], how="inner")
df["S"] = df["wind_mwh"] / df["load_mwh"]

# =============================
# FIT MODEL 0 ON ALL HISTORY
# =============================
model0 = smf.ols("cr ~ S * C(TSO)", data=df).fit()
print(model0.summary())

# =============================
# FUTURE QUARTERS 2021Q1–2046Q4
# =============================
future_q = pd.period_range(pd.Period(FUTURE_START, freq="Q"),
                           pd.Period(FUTURE_END, freq="Q"),
                           freq="Q")

# last observed quarter for growth baseline
last_q_wind = wind_q["quarter"].max()
last_q_load = load_q["quarter"].max()
last_q = max(last_q_wind, last_q_load)

# =============================
# OPTION B: forecast wind & load -> future S
# =============================
wind_f = make_future_series(wind_q, "wind_mwh", G_WIND, future_q=future_q, last_q=last_q)
load_f = make_future_series(load_q, "load_mwh", G_LOAD, future_q=future_q, last_q=last_q)

future = wind_f.merge(load_f, on=["quarter", "TSO"], how="inner")
future["S"] = future["wind_mwh"] / future["load_mwh"]

# =============================
# PREDICT CURTAILMENT
# =============================
future["cr_hat"] = model0.predict(future).clip(CR_FLOOR, CR_CAP)

out = future[["quarter", "TSO", "wind_mwh", "load_mwh", "S", "cr_hat"]].copy()
out = out.sort_values(["TSO", "quarter"]).reset_index(drop=True)

out.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
print(f"Quarter parsing OK (example): {curt['quarter'].iloc[0]}")
print(f"Baseline last_q used for growth: {last_q}")
