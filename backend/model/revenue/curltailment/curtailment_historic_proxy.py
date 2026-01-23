"""
Purpose
-------
Create a quarterly curtailment-rate dataset for 2015Q1..2025Q4 using ONLY a proxy model
based on wind generation and grid load.

We read the original curtailment source file "curtailment.csv" (quarterly, by TSO)
and compute the observed curtailment rate internally:

    cr_obs = curtailed_mwh / (curtailed_mwh + produced_mwh)

Those observed rates are used ONLY to fit/validate the proxy model.
The output series is fully proxy-generated.

Output (single file)
--------------------
curtailment_rate_quarterly_proxy_by_tso.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =============================================================================
# CONFIG
# =============================================================================
WIND_DAILY_FILE = "wind_data/wind_onshore_daily_by_tso.csv"
LOAD_MONTHLY_FILE = "grid_data/tso_grid_load_monthly_wide.csv"

# NEW: use the raw curtailment source file directly
CURTAILMENT_FILE = "curtailment.csv"

START_QUARTER = "2015Q1"
END_QUARTER = "2025Q4"

TSOS = ["TenneT", "50Hertz", "Amprion", "TransnetBW"]

LOAD_COL_MAP = {
    "TenneT": "load_mwh_tennet",
    "50Hertz": "load_mwh_50hertz",
    "Amprion": "load_mwh_amprion",
    "TransnetBW": "load_mwh_transnetbw",
}

# Physical bounds for curtailment rates
CR_FLOOR = 0.0
CR_CAP = 0.30

# Output (ONLY this file)
OUT_WIDE = "curtailment_rate_quarterly_proxy_by_tso.csv"


# =============================================================================
# HELPERS
# =============================================================================
def to_quarter(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("Q")


def wide_to_quarter_long(df: pd.DataFrame, date_col: str, value_cols: list[str], value_name: str) -> pd.DataFrame:
    """
    Convert wide daily/monthly data into long quarterly totals:
      quarter, TSO, <value_name>

    We sum within quarter because both wind and load are energy quantities (MWh).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["quarter"] = to_quarter(df[date_col])

    long = df.melt(id_vars="quarter", value_vars=value_cols, var_name="TSO", value_name=value_name)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.dropna(subset=["quarter", "TSO", value_name])

    return long.groupby(["quarter", "TSO"], as_index=False)[value_name].sum()


def parse_german_number(x) -> float:
    """
    Curtailment CSV uses German number format, e.g. '863,24' meaning 863.24.

    We also see cases with thousand separators ('.') in some exports.
    """
    if pd.isna(x):
        return float("nan")
    return float(str(x).replace(".", "").replace(",", "."))


def parse_quarter_label(q: str) -> pd.Period:
    """
    Convert labels like 'Q1 2015' into pandas Period('2015Q1').
    """
    q = str(q).strip()
    year = int(q[-4:])
    quarter = int(q[1])
    return pd.Period(year=year, quarter=quarter, freq="Q")


def build_observed_curtailment_rates(prod_q_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Build observed quarterly curtailment rates per TSO from:
    - wind production (quarterly totals) and
    - curtailment.csv (quarterly curtailed energy by TSO)

    Returns long format:
      quarter, TSO, cr_obs
    """
    curt = pd.read_csv(CURTAILMENT_FILE)

    if "Quarter" not in curt.columns:
        raise ValueError(f"curtailment.csv must contain a 'Quarter' column. Columns: {curt.columns.tolist()}")

    # Parse quarter labels from the curtailment sheet
    curt["quarter"] = curt["Quarter"].apply(parse_quarter_label)

    # Parse German numbers and convert GWh -> MWh (your file is in GWh)
    for tso in TSOS:
        if tso not in curt.columns:
            raise ValueError(f"curtailment.csv missing TSO column '{tso}'. Columns: {curt.columns.tolist()}")
        curt[tso] = curt[tso].apply(parse_german_number) * 1_000.0

    curt_q = curt[["quarter"] + TSOS].copy()

    # Merge production totals with curtailed energy (inner: only quarters we truly observe)
    df = prod_q_wide.merge(curt_q, on="quarter", how="inner", suffixes=("_prod", "_curt"))

    # Compute rates and return as long format
    rows = []
    for tso in TSOS:
        prod_col = f"{tso}_prod"
        curt_col = tso  # after merge, curtailment stays as tso name

        produced = pd.to_numeric(df[prod_col], errors="coerce").astype(float)
        curtailed = pd.to_numeric(df[curt_col], errors="coerce").astype(float)

        denom = produced + curtailed
        cr = np.where(denom > 0, curtailed / denom, np.nan)

        tmp = pd.DataFrame({"quarter": df["quarter"], "TSO": tso, "cr_obs": cr})
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["quarter", "TSO", "cr_obs"])
    out["cr_obs"] = out["cr_obs"].clip(CR_FLOOR, CR_CAP)
    return out


# =============================================================================
# BUILD QUARTERLY WIND / LOAD PANEL (2015..2025)
# =============================================================================
start_q = pd.Period(START_QUARTER, freq="Q")
end_q = pd.Period(END_QUARTER, freq="Q")

# Wind: daily -> quarterly totals (MWh)
wind = pd.read_csv(WIND_DAILY_FILE, parse_dates=["date"])
wind_q_long = wide_to_quarter_long(wind, "date", TSOS, "wind_mwh")

# We also keep a wide version of quarterly wind totals, because the observed-rate calculation
# needs the production totals per TSO in wide form.
wind_q_wide = (
    wind_q_long.pivot(index="quarter", columns="TSO", values="wind_mwh")
    .reset_index()
    .rename(columns={tso: f"{tso}_prod" for tso in TSOS})
)

# Load: monthly -> quarterly totals (MWh)
load = pd.read_csv(LOAD_MONTHLY_FILE)
load = load.rename(columns={v: k for k, v in LOAD_COL_MAP.items()})
load_q_long = wide_to_quarter_long(load, "date", TSOS, "load_mwh")

# Merge wind + load into a single modeling panel
panel = wind_q_long.merge(load_q_long, on=["quarter", "TSO"], how="inner")
panel["S"] = panel["wind_mwh"] / panel["load_mwh"]
panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["S"])

# Restrict to requested window
panel = panel.loc[(panel["quarter"] >= start_q) & (panel["quarter"] <= end_q)].copy()
wind_q_wide = wind_q_wide.loc[(wind_q_wide["quarter"] >= start_q) & (wind_q_wide["quarter"] <= end_q)].copy()

if panel.empty:
    raise RuntimeError(
        "Wind/load panel is empty in the requested period. "
        "Check file paths and that both wind and load cover 2015..2025."
    )

# =============================================================================
# BUILD OBSERVED CURTAILMENT RATES (for fitting only)
# =============================================================================
obs_long = build_observed_curtailment_rates(wind_q_wide)
obs_long = obs_long.loc[(obs_long["quarter"] >= start_q) & (obs_long["quarter"] <= end_q)].copy()

# Join observed rates onto the wind/load panel
panel = panel.merge(obs_long, on=["quarter", "TSO"], how="left")

train = panel.dropna(subset=["cr_obs"]).copy()
if len(train) < 20:
    raise RuntimeError(
        "Not enough observed curtailment-rate rows to fit proxy model. "
        "Check that curtailment.csv overlaps the wind/load period."
    )

# =============================================================================
# FIT PROXY MODEL + GENERATE PROXY CURTAILMENT SERIES
# =============================================================================
model = smf.ols("cr_obs ~ S * C(TSO)", data=train).fit()
print(model.summary())

panel["cr_proxy"] = np.clip(model.predict(panel).astype(float), CR_FLOOR, CR_CAP)

# =============================================================================
# WRITE FINAL OUTPUT (WIDE, PROXY ONLY)
# =============================================================================
out = (
    panel[["quarter", "TSO", "cr_proxy"]]
    .pivot(index="quarter", columns="TSO", values="cr_proxy")
    .reset_index()
)

out = out.rename(columns={tso: f"{tso}_curtailment_rate" for tso in TSOS})
out = out.sort_values("quarter").reset_index(drop=True)
out["quarter"] = out["quarter"].astype(str)

out.to_csv(OUT_WIDE, index=False)

print(f"\n[PROXY] Wrote proxy curtailment series -> {OUT_WIDE}")
print(f"[PROXY] Quarter range: {out['quarter'].iloc[0]} .. {out['quarter'].iloc[-1]}")
print(f"[PROXY] Observed fit rows used: {len(train):,}")
