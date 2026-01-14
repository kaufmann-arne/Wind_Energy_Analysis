import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- CONFIG ----------------
PANEL_FILE = "panel_monthly_prices.csv"
FORECAST_FILE = "forecast_merchant_vs_eeg_20y.csv"
HIST_END = "2025-12-31"
# ----------------------------------------


# ---------- Load ----------
panel = pd.read_csv(PANEL_FILE, parse_dates=["month"])
forecast = pd.read_csv(FORECAST_FILE, parse_dates=["month"])

# ---------- Split ----------
panel_hist = panel[panel["month"] <= HIST_END].copy()
future = forecast[forecast["month"] > HIST_END].copy()

# ---------- Metrics (merchant model only) ----------
actual = panel_hist["wind_capture_price_eur_mwh"]
pred_merchant = panel_hist["merchant_price_pred_eur_mwh"]

r2 = r2_score(actual, pred_merchant)
mae = mean_absolute_error(actual, pred_merchant)
rmse = np.sqrt(mean_squared_error(actual, pred_merchant))

print("\n=== Merchant wind price model accuracy ===")
print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.2f} €/MWh")
print(f"RMSE : {rmse:.2f} €/MWh")

# ---------- Combine for plotting ----------
plot_df = pd.concat(
    [
        panel_hist[[
            "month",
            "wind_capture_price_eur_mwh",
            "merchant_price_pred_eur_mwh",
            "eeg_supported_price_pred_eur_mwh",
        ]],
        future[[
            "month",
            "merchant_price_pred_eur_mwh",
            "eeg_supported_price_pred_eur_mwh",
        ]],
    ],
    ignore_index=True,
)

# ---------- Plot ----------
plt.figure(figsize=(12, 6))

# Actual (historical only)
plt.plot(
    panel_hist["month"],
    panel_hist["wind_capture_price_eur_mwh"],
    label="Actual wind capture price",
    linewidth=2,
)

# Merchant prediction
plt.plot(
    plot_df["month"],
    plot_df["merchant_price_pred_eur_mwh"],
    linestyle="--",
    label="Predicted merchant wind price",
)

# EEG-supported price
plt.plot(
    plot_df["month"],
    plot_df["eeg_supported_price_pred_eur_mwh"],
    linestyle=":",
    label="Predicted EEG-supported price",
)

# Vertical split line
plt.axvline(
    pd.Timestamp(HIST_END),
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Year")
plt.ylabel("€/MWh")
plt.title("Wind price comparison: actual vs merchant vs EEG-supported")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
