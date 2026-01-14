import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =============================
# CONFIG
# =============================
WIND_DAILY_FILE = "wind_onshore_daily_by_tso.csv"
LOAD_MONTHLY_FILE = "tso_grid_load_monthly_wide.csv"
CURTAIL_FILE = "curltailment_rate_quarterly_by_tso.csv"

TRAIN_END = "2020Q4"   # for backtest
CR_FLOOR = 0.0
CR_CAP = 0.30

TSOS = ["TenneT", "50Hertz", "Amprion", "TransnetBW"]
LOAD_COL_MAP = {
    "TenneT": "load_mwh_tennet",
    "50Hertz": "load_mwh_50hertz",
    "Amprion": "load_mwh_amprion",
    "TransnetBW": "load_mwh_transnetbw",
}

# =============================
# HELPERS
# =============================
def to_quarter(s):
    return pd.to_datetime(s).dt.to_period("Q").astype(str)

def wide_to_quarter_long(df, date_col, value_cols, value_name):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["quarter"] = to_quarter(df[date_col])
    long = df.melt(id_vars="quarter", value_vars=value_cols,
                   var_name="TSO", value_name=value_name)
    return long.groupby(["quarter", "TSO"], as_index=False)[value_name].sum()

# =============================
# LOAD & PREPROCESS
# =============================
# Wind
wind = pd.read_csv(WIND_DAILY_FILE)
wind_q = wide_to_quarter_long(wind, "date", TSOS, "wind_mwh")

# Load
load = pd.read_csv(LOAD_MONTHLY_FILE)
load = load.rename(columns={v: k for k, v in LOAD_COL_MAP.items()})
load_q = wide_to_quarter_long(load, "date", TSOS, "load_mwh")

# Curtailment
curt = pd.read_csv(CURTAIL_FILE)
records = []
for tso in TSOS:
    records.append(pd.DataFrame({
        "quarter": curt["quarter"],
        "TSO": tso,
        "cr": curt[f"{tso}_curtailment_rate"],
        "prod_mwh": curt[f"{tso}_production_mwh"],
        "curt_mwh": curt[f"{tso}_curtailed_mwh"],
    }))
curt_q = pd.concat(records, ignore_index=True)

# Merge
df = curt_q.merge(wind_q, on=["quarter", "TSO"]).merge(load_q, on=["quarter", "TSO"])

# Proxies
df["S"] = df["wind_mwh"] / df["load_mwh"]
df["log_wind"] = np.log(df["wind_mwh"].clip(lower=1))
df["S_logW"] = df["S"] * df["log_wind"]

# =============================
# BACKTEST SPLIT
# =============================
train = df[df["quarter"] <= TRAIN_END].copy()
test = df[df["quarter"] > TRAIN_END].copy()

# =============================
# MODELS
# =============================
models = {
    "M0_baseline": "cr ~ S * C(TSO)",
    "M1_interaction": "cr ~ S * C(TSO) + S_logW * C(TSO)",
}

results = []

for name, formula in models.items():
    model = smf.ols(formula, data=train).fit()

    for split, data in [("train", train), ("test", test)]:
        pred = model.predict(data).clip(CR_FLOOR, CR_CAP)

        for tso, g in data.assign(pred=pred).groupby("TSO"):
            rmse = np.sqrt(np.mean((g["cr"] - g["pred"])**2))
            mae = np.mean(np.abs(g["cr"] - g["pred"]))
            vol_err = (g["pred"] * g["prod_mwh"] - g["curt_mwh"]).sum()

            results.append({
                "model": name,
                "split": split,
                "TSO": tso,
                "RMSE": rmse,
                "MAE": mae,
                "Curt_MWh_Error": vol_err,
            })

    print(f"\n=== {name} ===")
    print(model.summary())

# =============================
# RESULTS
# =============================
res = pd.DataFrame(results)
res.to_csv("curtailment_model_comparison.csv", index=False)

print("\n=== MODEL COMPARISON (TEST SET) ===")
print(res[res["split"] == "test"].sort_values(["model", "TSO"]))
