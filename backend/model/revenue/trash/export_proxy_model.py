import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =============================
# FILES
# =============================
WIND_DAILY_FILE = "wind_onshore_daily_by_tso.csv"
LOAD_MONTHLY_FILE = "tso_grid_load_monthly_wide.csv"
CURTAIL_FILE = "curltailment_rate_quarterly_by_tso.csv"
EXPORT_FILE = "tso_net_exports_hourly_wide.csv"

TSOS = ["TenneT", "50Hertz", "Amprion", "TransnetBW"]

LOAD_COL_MAP = {
    "TenneT": "load_mwh_tennet",
    "50Hertz": "load_mwh_50hertz",
    "Amprion": "load_mwh_amprion",
    "TransnetBW": "load_mwh_transnetbw",
}

TRAIN_END = "2020Q4"
CR_CAP = 0.30

# =============================
def to_quarter(s):
    return pd.to_datetime(s).dt.to_period("Q").astype(str)

def wide_to_quarter_long(df, date_col, value_cols, value_name):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["quarter"] = to_quarter(df[date_col])

    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long = df.melt(id_vars="quarter", value_vars=value_cols,
                   var_name="TSO", value_name=value_name)
    long = long.dropna(subset=[value_name])

    return long.groupby(["quarter", "TSO"], as_index=False)[value_name].sum()

def load_curtailment_long(path):
    curt = pd.read_csv(path)
    recs = []
    for tso in TSOS:
        recs.append(pd.DataFrame({
            "quarter": curt["quarter"].astype(str),
            "TSO": tso,
            "cr": curt[f"{tso}_curtailment_rate"],
            "prod_mwh": curt[f"{tso}_production_mwh"],
            "curt_mwh": curt[f"{tso}_curtailed_mwh"],
        }))
    return pd.concat(recs, ignore_index=True)

def build_export_regime(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["quarter"] = to_quarter(df["datetime"])

    tso_cols = [c for c in df.columns if c.startswith("netexp_")]

    # Germany-wide export stress (sum over TSOs)
    df["netexp_DE"] = df[tso_cols].sum(axis=1)

    # P95 of hourly exports in each quarter
    reg = df.groupby("quarter")["netexp_DE"].quantile(0.95).reset_index()
    reg = reg.rename(columns={"netexp_DE": "ExportStress"})
    return reg

def metrics(df):
    out = []
    for tso, g in df.groupby("TSO"):
        rmse = np.sqrt(np.mean((g["cr"] - g["pred"])**2))
        mae = np.mean(np.abs(g["cr"] - g["pred"]))
        vol_err = (g["pred"] * g["prod_mwh"] - g["curt_mwh"]).sum()
        out.append([tso, rmse, mae, vol_err])
    return pd.DataFrame(out, columns=["TSO", "RMSE", "MAE", "Curt_MWh_Error"])

# =============================
def main():
    # ---- Build S = wind / load
    wind = pd.read_csv(WIND_DAILY_FILE)
    wind_q = wide_to_quarter_long(wind, "date", TSOS, "wind_mwh")

    load = pd.read_csv(LOAD_MONTHLY_FILE)
    load = load.rename(columns={v: k for k,v in LOAD_COL_MAP.items()})
    load_q = wide_to_quarter_long(load, "date", TSOS, "load_mwh")

    curt = load_curtailment_long(CURTAIL_FILE)
    exp_reg = build_export_regime(EXPORT_FILE)

    df = curt.merge(wind_q, on=["quarter","TSO"])\
             .merge(load_q, on=["quarter","TSO"])\
             .merge(exp_reg, on="quarter")

    df["S"] = df["wind_mwh"] / df["load_mwh"]

    train = df[df["quarter"] <= TRAIN_END].copy()
    test  = df[df["quarter"] > TRAIN_END].copy()

    # ---- Model 2
    formula = "cr ~ S * C(TSO) + ExportStress"
    model = smf.ols(formula, data=train).fit(cov_type="HC3")

    print("\n=== MODEL 2: Wind share + export regime ===")
    print(model.summary())

    train["pred"] = model.predict(train).clip(0, CR_CAP)
    test["pred"]  = model.predict(test).clip(0, CR_CAP)

    print("\n=== TRAIN METRICS ===")
    print(metrics(train))

    print("\n=== TEST METRICS ===")
    print(metrics(test))

    # Compare to Model 0 results you already saved
    train.to_csv("curtailment_model2_train.csv", index=False)
    test.to_csv("curtailment_model2_test.csv", index=False)

    print("\nSaved: curtailment_model2_train.csv, curtailment_model2_test.csv")

if __name__ == "__main__":
    main()
