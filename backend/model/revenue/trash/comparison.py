import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config (edit if needed)
# -----------------------------
NT_FILE = "netztransparenz_wind_onshore_monthly.csv"
SMARD_FILE = "smard_calc_wind_onshore_monthly.csv"

OUTPUT_CSV = "comparison_nt_vs_smard_monthly.csv"
PLOT_PREFIX = "comparison_nt_vs_smard"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NT_PATH = os.path.join(BASE_DIR, NT_FILE)
SMARD_PATH = os.path.join(BASE_DIR, SMARD_FILE)
OUT_PATH = os.path.join(BASE_DIR, OUTPUT_CSV)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

def main():
    nt = pd.read_csv(NT_PATH, parse_dates=["date"])
    sm = pd.read_csv(SMARD_PATH, parse_dates=["date"])

    # Expected columns
    if "wind_onshore_eur_mwh" not in nt.columns:
        raise ValueError("Netztransparenz file must contain column: wind_onshore_eur_mwh")
    if "wind_onshore_eur_mwh_calc" not in sm.columns:
        raise ValueError("SMARD calc file must contain column: wind_onshore_eur_mwh_calc")

    # Merge on date
    df = nt.merge(sm, on="date", how="inner")

    # Rename for clarity
    df = df.rename(columns={
        "wind_onshore_eur_mwh": "nt_eur_mwh",
        "wind_onshore_eur_mwh_calc": "smard_eur_mwh"
    })

    # Drop missing
    df = df.dropna(subset=["nt_eur_mwh", "smard_eur_mwh"]).sort_values("date")

    # Errors
    df["error"] = df["smard_eur_mwh"] - df["nt_eur_mwh"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = np.where(df["nt_eur_mwh"] != 0, df["error"] / df["nt_eur_mwh"], np.nan)
    df["abs_pct_error"] = df["pct_error"].abs()

    # Metrics
    y_true = df["nt_eur_mwh"].to_numpy()
    y_pred = df["smard_eur_mwh"].to_numpy()

    bias = float(np.mean(df["error"]))
    mae = float(np.mean(df["abs_error"]))
    rmse = float(np.sqrt(np.mean(df["error"] ** 2)))
    mape = float(np.mean(df["abs_pct_error"]) * 100.0)

    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(df) > 1 else float("nan")
    r2 = float(r2_score(y_true, y_pred))

    # Print summary
    print("=== Netztransparenz vs SMARD-calculated wind value (monthly) ===")
    print(f"Months compared: {len(df)}")
    print(f"Date range: {df['date'].min().date()}  →  {df['date'].max().date()}")
    print()
    print("Errors defined as: SMARD - Netztransparenz (€/MWh)")
    print(f"Bias (mean error): {bias: .3f} €/MWh")
    print(f"MAE:               {mae: .3f} €/MWh")
    print(f"RMSE:              {rmse: .3f} €/MWh")
    print(f"MAPE:              {mape: .2f} %")
    print(f"Correlation:       {corr: .4f}")
    print(f"R²:                {r2: .4f}")

    # Save detailed comparison
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved detailed comparison to: {OUTPUT_CSV}")

    # -----------------------------
    # Plots
    # -----------------------------
    # 1) Time series overlay
    plt.figure()
    plt.plot(df["date"], df["nt_eur_mwh"], label="Netztransparenz (MW Wind an Land)")
    plt.plot(df["date"], df["smard_eur_mwh"], label="SMARD calc (wind-weighted DA)")
    plt.xlabel("Date")
    plt.ylabel("€/MWh")
    plt.title("Monthly wind market value: Netztransparenz vs SMARD calculation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{PLOT_PREFIX}_timeseries.png"), dpi=200)

    # 2) Scatter
    plt.figure()
    plt.scatter(df["nt_eur_mwh"], df["smard_eur_mwh"])
    # 45-degree line
    minv = min(df["nt_eur_mwh"].min(), df["smard_eur_mwh"].min())
    maxv = max(df["nt_eur_mwh"].max(), df["smard_eur_mwh"].max())
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("Netztransparenz €/MWh")
    plt.ylabel("SMARD calc €/MWh")
    plt.title("Scatter: SMARD calc vs Netztransparenz (45° line = perfect match)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{PLOT_PREFIX}_scatter.png"), dpi=200)

    # 3) Error over time
    plt.figure()
    plt.plot(df["date"], df["error"])
    plt.axhline(0)
    plt.xlabel("Date")
    plt.ylabel("SMARD - Netztransparenz (€/MWh)")
    plt.title("Monthly error over time")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{PLOT_PREFIX}_error.png"), dpi=200)

    print("Saved plots:")
    print(f" - {PLOT_PREFIX}_timeseries.png")
    print(f" - {PLOT_PREFIX}_scatter.png")
    print(f" - {PLOT_PREFIX}_error.png")

if __name__ == "__main__":
    main()
