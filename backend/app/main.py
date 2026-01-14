from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
from pathlib import Path
import sys

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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request/response schemas
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

class CalcResponse(BaseModel):
    ok: bool
    npv_eur: float
    irr: float
    discount_rate_used: float
    yearly: List[dict]

@app.post("/api/calc", response_model=CalcResponse)
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

        yearly_df = result["yearly_df"].copy()
        if "year_start" in yearly_df.columns:
            yearly_df["year_start"] = yearly_df["year_start"].astype(str)

        irr_val = result["irr"]
        irr_val = float(irr_val) if irr_val == irr_val else float("nan")  # handle NaN

        return {
            "ok": True,
            "npv_eur": float(result["npv_eur"]),
            "irr": irr_val,
            "discount_rate_used": float(result["discount_rate_used"]),
            "yearly": yearly_df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
