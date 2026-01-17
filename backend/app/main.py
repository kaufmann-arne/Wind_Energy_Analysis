from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import math


# --- Make sure Python can import from backend/model ---
# project/backend/app/main.py -> project/backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

# Now we can import your script
from profit import power_to_profit, FORECAST_MONTHS

app = FastAPI()

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/")
def root():
    return {"ok": True, "message": "Backend running. Go to /docs for API docs."}

# Allow the Vite dev server to call the API from the browser during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request schema
# -------------------------
class CalcRequest(BaseModel):
    # Settings from the UI
    n_turbines: int = Field(ge=1)
    hub_height_m: int = Field(ge=1)
    turbine_type_id: int = Field(ge=0, le=2)
    equity_eur: int = Field(ge=0)

    # Revenue module required keys
    tso_id: int = Field(ge=1)
    eeg_on: bool
    cod_date: str  # "YYYY-MM-DD"
    manual_eeg_strike: Optional[float] = None

    # For now: constant monthly MWh (easy to wire first)
    mwh_constant: float = Field(gt=0)


def to_jsonable(obj: Any):
    """Convert pandas/numpy objects into JSON-serializable Python types.
    Also replaces NaN/Inf with None (JSON-compliant).
    """
    # pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        # Convert datetime columns to strings
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        # Replace inf/-inf with NaN, then NaN -> None
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)

        return df.to_dict(orient="records")

    # pandas Series
    if isinstance(obj, pd.Series):
        s = obj.copy()

        # If index is datetime, return list of {date, value}
        if pd.api.types.is_datetime64_any_dtype(s.index):
            out = []
            for idx, val in s.items():
                if pd.isna(val) or val in (np.inf, -np.inf):
                    v = None
                else:
                    v = float(val)
                out.append({"date": str(idx), "value": v})
            return out

        # otherwise just list values
        out = []
        for x in s.to_list():
            if pd.isna(x) or x in (np.inf, -np.inf):
                out.append(None)
            else:
                out.append(float(x))
        return out

    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return str(obj)

    # numpy scalars/arrays
    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return val

    if isinstance(obj, np.ndarray):
        # Convert elements too (in case the array has NaN)
        return [to_jsonable(x) for x in obj.tolist()]

    # plain Python float NaN/Inf
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj

    # dict / list / tuple
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    return obj

def scrub_nonfinite(obj: Any):
    """Recursively replace NaN/Inf/-Inf with None in already JSON-like objects."""
    # dict
    if isinstance(obj, dict):
        return {k: scrub_nonfinite(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [scrub_nonfinite(v) for v in obj]

    # numpy scalar that may still exist
    if isinstance(obj, np.generic):
        return scrub_nonfinite(obj.item())

    # float NaN/Inf
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj

    return obj

def pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None




@app.post("/api/calc")
def calc(req: CalcRequest):
    try:
        # Build monthly MWh series for 20 years
        cod = pd.Timestamp(req.cod_date)
        cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

        dates = pd.date_range(cod, periods=FORECAST_MONTHS, freq="MS")
        mwh_series = pd.Series(float(req.mwh_constant), index=dates)

        result = power_to_profit(
            mwh_monthly_20y=mwh_series,
            turbine_type_id=req.turbine_type_id,
            n_turbines=req.n_turbines,
            hub_height_m=req.hub_height_m,
            equity_eur=float(req.equity_eur),
            revenue_kwargs={
                "tso_id": int(req.tso_id),
                "eeg_on": 1 if req.eeg_on else 0,
                "manual_eeg_strike": req.manual_eeg_strike,
                "cod_date": req.cod_date,
            },
        )

        monthly = result["monthly_revenue_df"].copy()
        yearly = result["yearly_df"].copy()

        # Ensure project_year exists (it should, from your profit.py logic)
        if "project_year" not in monthly.columns:
            raise ValueError("monthly_revenue_df is missing 'project_year'")

        # Choose a reasonable MWh column (depends on your revenue module output)
        mwh_col = pick_first_col(monthly, ["mwh_gross", "mwh", "mwh_delivered"])
        if not mwh_col:
            raise ValueError("monthly_revenue_df has none of: mwh_gross, mwh, mwh_delivered")

        # Weighted average helper (for market price)
        def wavg(group: pd.DataFrame, value_col: str, weight_col: str):
            w = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0)
            v = pd.to_numeric(group[value_col], errors="coerce")
            denom = float(w.sum())
            if denom <= 0:
                return np.nan
            return float((v * w).sum() / denom)

        # Aggregate monthly -> yearly drivers
        agg = (
            monthly.groupby("project_year", as_index=False)
            .apply(lambda g: pd.Series({
                "mwh_gross": float(pd.to_numeric(g[mwh_col], errors="coerce").fillna(0.0).sum()),
                "market_price_eur_per_mwh": wavg(g, "p_market", mwh_col) if "p_market" in g.columns else np.nan,
                "cf": float(pd.to_numeric(g["cf"], errors="coerce").mean()) if "cf" in g.columns else np.nan,
                "cr": float(pd.to_numeric(g["cr"], errors="coerce").mean()) if "cr" in g.columns else np.nan,
            }))
        )

        # Join with yearly opex/debt/profit
        out = yearly.merge(agg, on="project_year", how="left")

        # Build the exact columns you asked for
        table_df = pd.DataFrame({
            "year": out["project_year"],
            "mwh_gross": out["mwh_gross"],
            "market_price_electricity": out["market_price_eur_per_mwh"],
            "cf": out["cf"],
            "cr": out["cr"],
            "opex": out["annual_opex_eur"],
            "debt_service": out["debt_service_eur"],
            "profit": out["profit_after_opex_and_debt_eur"],
        })

        # Convert NaN/Inf -> None for JSON
        table_df = table_df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(table_df), None)

        payload = {
            "ok": True,
            "npv_eur": float(result["npv_eur"]),
            "irr": None if (float(result["irr"]) != float(result["irr"])) else float(result["irr"]),
            "discount_rate_used": float(result["discount_rate_used"]),
            "table": table_df.to_dict(orient="records"),
        }

        # Ensure strict JSON (no NaN)
        json.dumps(payload, allow_nan=False)

        return payload


    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
