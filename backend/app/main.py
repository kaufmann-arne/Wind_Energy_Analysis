from __future__ import annotations

# Standard library imports
import io
import base64
import os
import json
import math
import sys
import hashlib
import traceback
from pathlib import Path
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from time import perf_counter

# Third-party HTTP client used for outbound email API calls
import requests

# Matplotlib is used only for server-side chart rendering.
# "Agg" backend ensures headless rendering without a display (suitable for containers/servers).
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ReportLab is used for deterministic PDF generation (headers, tables, images, pagination).
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Numerical + dataframe utilities for forecast and financial model outputs
import numpy as np
import pandas as pd

# Web service framework and request/response schemas
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# PATH SETUP
# project/backend/app/main.py -> project/backend
# ============================================================
# Determine backend root directory based on the location of this file:
#   project/backend/app/main.py
#   parents[1] -> project/backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]   # .../project/backend

# Extend module search path to allow importing backend/model (profit model code).
# This avoids needing a package install step in lightweight deployments.
MODEL_DIR = BACKEND_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

# ML subproject root:
# backend/wind-power-climate-ml/
# The ML code is referenced via "ml.*" imports and requires its src folder on sys.path.
ML_ROOT = BACKEND_ROOT / "wind-power-climate-ml"
ML_SRC = ML_ROOT / "src"
sys.path.insert(0, str(ML_SRC))  # enables: from ml.production_forecast import ...

# Local assets located alongside this file (e.g., report header logo).
THIS_DIR = Path(__file__).resolve().parent
LOGO_PATH = THIS_DIR / "windsideanalytics.png"


# ============================================================
# Imports that depend on PATH SETUP
# ============================================================
# Profit model entrypoint and configuration constants.
from profit_mc import power_to_profit, FORECAST_MONTHS  # noqa: E402

# Forecast helpers from the ML subproject. These build historical monthly energy series
# and sample a representative production path using Monte Carlo from historical energy.
from ml.production_forecast import (  # noqa: E402
    build_hist_monthly_energy_lookup,
    forecast_monthly_representative_path_from_hist_energy,
)


# ============================================================
# ENV SWITCH: local vs server
# ============================================================
# Environment switch used to tune CORS and server bind defaults.
# "local/dev/development" enables restrictive localhost origins.
APP_ENV = os.getenv("APP_ENV", "server").lower()
IS_LOCAL = APP_ENV in ("local", "dev", "development")

# Bind/configuration for running locally vs. server.
LOCAL_PORT = int(os.getenv("LOCAL_PORT", "5137"))
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# FastAPI application instance.
app = FastAPI()


@app.get("/favicon.ico")
def favicon():
    # Avoid unnecessary 404 noise in browser consoles and logs.
    return Response(status_code=204)


@app.get("/")
def root():
    # Basic health/status endpoint for smoke tests and deployments.
    return {
        "ok": True,
        "env": APP_ENV,
        "message": "Backend running. Go to /docs for API docs.",
    }


# ============================================================
# CORS
# ============================================================
# Local development typically runs frontend and backend on separate ports.
# In production, origins should be tightened once deployment domains are known.
if IS_LOCAL:
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5137",
        "http://127.0.0.1:5137",
    ]
else:
    allow_origins = ["*"]  # tighten later if needed

# Apply CORS middleware for cross-origin API calls from the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Forecast data paths (env-overridable)
# ============================================================
# Resolve forecast inputs with environment overrides to support:
# - Local paths
# - Mounted volumes in containers
# - Different model artifact versions per environment
DEFAULT_ERA5_DIR = Path(
    os.getenv("ERA5_DIR", str(ML_ROOT / "data" / "raw" / "era5"))
)
DEFAULT_TURBINE_CURVES = Path(
    os.getenv("TURBINE_CURVES_JSON", str(ML_ROOT / "data" / "turbine_power_curves.json"))
)
DEFAULT_MODEL_DIR = Path(
    os.getenv("FORECAST_MODEL_DIR", str(ML_ROOT / "LGBM_Model" / "model_artifacts"))
)

# Constant-production fallback used when forecast mode is not selected/available.
DEFAULT_MWH_PER_TURBINE_PER_MONTH = 1200.0


# Optional sanity checks (recommended during development)
# Fail fast if critical forecast/model inputs are missing.
for p, label in [
    (DEFAULT_ERA5_DIR, "ERA5 directory"),
    (DEFAULT_TURBINE_CURVES, "Turbine curves JSON"),
    (DEFAULT_MODEL_DIR, "Forecast model_artifacts"),
]:
    if not p.exists():
        raise RuntimeError(f"{label} not found: {p}")


# ============================================================
# Request schema (unified)
# ============================================================

class CalcRequest(BaseModel):
    # Ignore unknown fields to keep the API resilient to frontend schema evolution.
    model_config = ConfigDict(extra="ignore")

    # Location + meta
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    email: Optional[str] = None

    # UI settings
    # Minimum of 1 turbine and a positive hub height ensures valid model inputs.
    n_turbines: int = Field(default=5, ge=1)
    hub_height_m: int = Field(default=160, ge=1)

    # Profit model turbine selection (numeric)
    # Constrained to supported model ids.
    turbine_type_id: int = Field(default=1, ge=0, le=2)

    # Forecast speed:
    # True => fast forecast; False => slow forecast
    # None allows backend to infer defaults.
    fast_mode: Optional[bool] = None

    # Equity input for the financial model (EUR).
    equity_eur: int = Field(default=1_000_000, ge=0)

    # Revenue settings
    # tso_id selects price zone/path assumptions; bounded to known options.
    tso_id: int = Field(default=1, ge=0, le=3)
    eeg_on: bool = True
    # Commercial operation date, "YYYY-MM-DD". Normalized to month-start internally.
    cod_date: str  # "YYYY-MM-DD"


# ============================================================
# Email helper
# ============================================================
def _parse_iso_dt(ts: str) -> Optional[datetime]:
    """
    Parse ISO timestamps with tolerant handling of 'Z' suffix and missing tz info.

    Returns:
        UTC-aware datetime on success, or None on parse failures.
    """
    if not ts:
        return None
    try:
        # Convert "Z" into an explicit UTC offset for fromisoformat.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        # Assume UTC if tz information is absent.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _val(d: dict, key: str, placeholder: str) -> Any:
    """
    Retrieve a dictionary value with placeholder fallback for None/empty values.

    This supports templated PDF fields where unresolved values should stay bracketed.
    """
    v = d.get(key, None)
    return placeholder if v is None or v == "" else v


def _fmt_num(x: Any, placeholder: str) -> str:
    """
    Format numeric values into compact, human-readable strings.
    - >= 1,000,000 as millions (e.g., 1.23m)
    - >= 1,000 with thousands separators
    - else with 2 decimals

    Non-numeric values and missing values return the placeholder.
    """
    if x is None or x == "" or x == placeholder:
        return str(placeholder)
    try:
        if isinstance(x, (int, float)):
            if abs(x) >= 1_000_000:
                return f"{x/1_000_000:.2f}m"
            if abs(x) >= 1_000:
                return f"{x:,.0f}"
            return f"{x:.2f}"
        return str(x)
    except Exception:
        return str(placeholder)


def _fmt_pct(x: Any, placeholder: str) -> str:
    """
    Format percentages with tolerant input handling.
    - Values in [-1, 1] are interpreted as fractions and converted to percent.
    - Otherwise treated as already-percent values.

    Returns placeholder for missing/invalid values.
    """
    if x is None or x == "" or x == placeholder:
        return str(placeholder)
    try:
        v = float(x)
        if abs(v) <= 1.0:
            v *= 100.0
        return f"{v:.2f}%"
    except Exception:
        return str(placeholder)


def _fmt_yesno(x: Any, placeholder: str = "[EEG_SUPPORT]") -> str:
    """
    Normalize a value into "Yes"/"No" strings.
    Useful for PDF display regardless of whether the input is boolean or a string.
    """
    if x is None or x == "":
        return placeholder
    if isinstance(x, bool):
        return "Yes" if x else "No"
    s = str(x).strip().lower()
    if s in {"true", "yes", "1", "y"}:
        return "Yes"
    if s in {"false", "no", "0", "n"}:
        return "No"
    return placeholder


def _today_str() -> str:
    """
    Resolve today's date in ISO format.
    Placeholder returned if any unexpected runtime error occurs.
    """
    try:
        return date.today().isoformat()
    except Exception:
        return "[REPORT_DATE]"


def _short_label(s: str, max_len: int = 14) -> str:
    """
    Create a shortened label for chart/table text.
    Uses ellipsis when text exceeds max_len.
    """
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# ============================
# Energy adapters
# ============================

def derive_monthly_and_annual_energy(
    energy_forecast: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert raw forecast rows into month-start and year aggregates.

    Input monthly:
      [{"timestamp":"...Z","energy_kwh":...}, ...]

    Returns:
      monthly_mwh: [{"month_start":"YYYY-MM-01","mwh":...}, ...]
      annual_mwh:  [{"year":YYYY,"mwh":...}, ...]
    """
    if not energy_forecast:
        return [], []

    # Parse timestamps and convert kWh -> MWh. Skip malformed rows robustly.
    rows: List[Tuple[datetime, float]] = []
    for r in energy_forecast:
        dt = _parse_iso_dt(str(r.get("timestamp", "")))
        if dt is None:
            continue
        try:
            kwh = float(r.get("energy_kwh"))
        except Exception:
            continue
        rows.append((dt, kwh / 1000.0))

    rows.sort(key=lambda x: x[0])
    if not rows:
        return [], []

    # Store month-start strings to align with MS-series conventions.
    monthly_mwh: List[Dict[str, Any]] = []
    for dt, mwh in rows:
        month_start = f"{dt.year:04d}-{dt.month:02d}-01"
        monthly_mwh.append({"month_start": month_start, "mwh": mwh})

    # Aggregate annual sums.
    by_year: Dict[int, float] = {}
    for dt, mwh in rows:
        by_year[dt.year] = by_year.get(dt.year, 0.0) + mwh
    annual_mwh = [{"year": y, "mwh": by_year[y]} for y in sorted(by_year.keys())]

    return monthly_mwh, annual_mwh


def derive_aep_and_capacity_factor(
    monthly_mwh: List[Dict[str, Any]],
    rated_power_mw: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute high-level energy KPIs.

    AEP:
        Average annual energy in MWh.
        Prefer complete calendar years; if only partial years exist, scale each partial year
        to an annual equivalent and then average.

    Capacity factor (CF):
        CF [%] = AEP / (rated_power_mw * 8760) * 100

    Returns:
        (aep_mwh_per_year, capacity_factor_pct) or (None, None) if inputs are insufficient.
    """
    if not monthly_mwh or not rated_power_mw:
        return None, None

    by_year_sum: Dict[int, float] = {}
    by_year_months: Dict[int, set] = {}

    # Collect annual totals and track months observed per year to detect completeness.
    for r in monthly_mwh:
        ms = str(r.get("month_start", ""))
        if len(ms) < 7 or not ms[:4].isdigit():
            continue
        y = int(ms[:4])
        m = int(ms[5:7]) if ms[5:7].isdigit() else None
        try:
            v = float(r.get("mwh"))
        except Exception:
            continue

        by_year_sum[y] = by_year_sum.get(y, 0.0) + v
        if m is not None:
            by_year_months.setdefault(y, set()).add(m)

    if not by_year_sum:
        return None, None

    # Prefer complete years for AEP if available.
    full_years = [y for y in by_year_months if len(by_year_months[y]) == 12]
    if full_years:
        aep = sum(by_year_sum[y] for y in full_years) / float(len(full_years))
    else:
        # Scale partial years to 12-month equivalents.
        scaled = []
        for y in sorted(by_year_sum.keys()):
            m_count = max(1, len(by_year_months.get(y, set())))
            scaled.append(by_year_sum[y] * (12.0 / m_count))
        aep = sum(scaled) / float(len(scaled)) if scaled else None

    cf = (aep / (rated_power_mw * 8760.0)) * 100.0 if (aep is not None and rated_power_mw) else None
    return aep, cf


# ============================
# Executive assessment helpers (NEW)
# ============================

def _safe_float(x: Any) -> Optional[float]:
    """Best-effort float conversion. Returns None for empty/invalid values."""
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _fmt_pct_from_frac(x: Any) -> str:
    """
    Format a value as a percent string.
    Treat values in a plausible fraction range as fractions (<= ~1.2) and convert to percent.
    """
    v = _safe_float(x)
    if v is None:
        return "n/a"
    if abs(v) <= 1.2:
        v *= 100.0
    return f"{v:.2f}%"


def _fmt_eur(x: Any) -> str:
    """
    Format EUR values with magnitude abbreviations (bn/m) and thousands separators.
    Returns "n/a" when unavailable.
    """
    v = _safe_float(x)
    if v is None:
        return "n/a"
    if abs(v) >= 1_000_000_000:
        return f"€{v/1_000_000_000:.2f}bn"
    if abs(v) >= 1_000_000:
        return f"€{v/1_000_000:.2f}m"
    if abs(v) >= 1_000:
        return f"€{v:,.0f}"
    return f"€{v:.0f}"


def _get_final_summary(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize payload["final_summary_table"] into a dictionary keyed by metric name.

    Supported input shapes:
      - dict: {metric_name: {mean,p10,p50,p90}, ...}
      - list: [{"metric": ..., "mean": ..., "p10": ..., "p50": ..., "p90": ...}, ...]

    Returns:
      { metric_name: {"mean":..,"p10":..,"p50":..,"p90":..}, ...}
    """
    tbl = payload.get("final_summary_table")
    if tbl is None:
        return {}

    if isinstance(tbl, dict):
        out = {}
        for k, v in tbl.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out

    if isinstance(tbl, list):
        out: Dict[str, Dict[str, Any]] = {}
        for row in tbl:
            if not isinstance(row, dict):
                continue
            # Allow for different column conventions.
            metric = row.get("metric") or row.get("Metric") or row.get("name")
            if not metric:
                continue
            out[str(metric)] = {
                "mean": row.get("mean"),
                "p10": row.get("p10"),
                "p50": row.get("p50"),
                "p90": row.get("p90"),
            }
        return out

    return {}


def _derive_aep_cf_from_monthly(
    monthly: List[Tuple[datetime, float]],
    park_mw: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute AEP and capacity factor fraction from datetime-indexed monthly tuples.

    Returns:
      (aep_mwh_per_year, capacity_factor_fraction)
    """
    if not monthly:
        return None, None
    park_mw_f = _safe_float(park_mw)
    if park_mw_f is None or park_mw_f <= 0:
        return None, None

    by_year_sum: Dict[int, float] = {}
    by_year_months: Dict[int, set] = {}

    # Aggregate monthly totals per year and track observed months to assess completeness.
    for dt, mwh in monthly:
        if not isinstance(dt, datetime):
            continue
        v = _safe_float(mwh)
        if v is None:
            continue
        y = dt.year
        by_year_sum[y] = by_year_sum.get(y, 0.0) + v
        by_year_months.setdefault(y, set()).add(dt.month)

    if not by_year_sum:
        return None, None

    full_years = [y for y in by_year_months if len(by_year_months[y]) == 12]
    if full_years:
        aep = sum(by_year_sum[y] for y in full_years) / float(len(full_years))
    else:
        # Scale partial years to annual equivalents.
        scaled = []
        for y in sorted(by_year_sum.keys()):
            m_count = max(1, len(by_year_months.get(y, set())))
            scaled.append(by_year_sum[y] * (12.0 / m_count))
        aep = sum(scaled) / float(len(scaled)) if scaled else None

    if aep is None:
        return None, None

    cf_frac = aep / (park_mw_f * 8760.0)
    return aep, cf_frac


def _variability_from_annual(
    annual: List[Tuple[int, float]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute inter-annual variability statistics for annual MWh series.

    Returns:
      (min, mean, max, coefficient_of_variation)
    """
    vals: List[float] = []
    for _, mwh in annual or []:
        v = _safe_float(mwh)
        if v is not None:
            vals.append(v)

    if not vals:
        return None, None, None, None

    mean_a = sum(vals) / len(vals)
    min_a = min(vals)
    max_a = max(vals)

    # CV requires at least two samples and non-zero mean.
    if len(vals) >= 2 and mean_a:
        var = sum((x - mean_a) ** 2 for x in vals) / (len(vals) - 1)
        std = var ** 0.5
        cv = std / mean_a
    else:
        cv = None

    return min_a, mean_a, max_a, cv


def _location_assessment_text(
    payload: Dict[str, Any],
    monthly: List[Tuple[datetime, float]],
    annual: List[Tuple[int, float]],
    park_mw: float,
) -> str:
    """
    Create a short, dynamic executive assessment based ONLY on payload values.

    Notes:
      - No external lookups or assumptions beyond what is present in payload.
      - Returned string targets ReportLab Paragraph with basic HTML support (e.g., <b>).
    """

    lat = payload.get("latitude", None)
    lon = payload.get("longitude", None)

    # --- Key inputs from payload ---
    # Economic outputs are stored as quantile dicts where available.
    npv = payload.get("npv_eur") or {}
    irr = payload.get("irr") or {}
    # Prefer explicit wacc if present, otherwise fall back to discount rate used.
    wacc = _safe_float(payload.get("wacc")) or _safe_float(payload.get("discount_rate_used"))

    # Risk signals may be embedded in final_summary_table; normalize to a dictionary.
    summary = _get_final_summary(payload)
    prob_npv_lt0 = (summary.get("Prob(NPV < 0)", {}) or {}).get("mean", None)

    # Try to find any "Prob(IRR < WACC...)" metric variant.
    prob_irr_lt_wacc = None
    for k, v in summary.items():
        if isinstance(k, str) and k.strip().startswith("Prob(IRR < WACC"):
            prob_irr_lt_wacc = (v or {}).get("mean", None)
            break

    # --- Derived energy metrics from payload series ---
    aep, aep_cf = _derive_aep_cf_from_monthly(monthly, park_mw)  # aep_cf fraction
    min_a, mean_a, max_a, cv_a = _variability_from_annual(annual)

    # ------------------------------
    # 1) Energy quality statement
    # ------------------------------
    if aep_cf is None:
        cf_sentence = "No reliable capacity-factor estimate could be derived from the monthly production series."
    else:
        # Thresholds reflect typical onshore/offshore capacity factor ranges for screening-level statements.
        if aep_cf >= 0.35:
            cf_sentence = f"The production profile indicates a <b>very strong</b> wind resource with an average capacity factor of {_fmt_pct_from_frac(aep_cf)}."
        elif aep_cf >= 0.25:
            cf_sentence = f"The site shows a <b>solid</b> wind resource with an average capacity factor of {_fmt_pct_from_frac(aep_cf)}."
        elif aep_cf >= 0.18:
            cf_sentence = f"The wind resource appears <b>moderate</b>, with an average capacity factor of {_fmt_pct_from_frac(aep_cf)}."
        else:
            cf_sentence = f"The wind resource appears <b>weak</b>, with an average capacity factor of {_fmt_pct_from_frac(aep_cf)}."

    if aep is not None:
        aep_sentence = f"This corresponds to an average annual energy of approximately <b>{_fmt_num(aep,'n/a')} MWh</b> for a <b>{_fmt_num(park_mw,'n/a')} MW</b> park."
    else:
        aep_sentence = f"Park size in the payload is <b>{_fmt_num(park_mw,'n/a')} MW</b>, but average annual energy could not be derived."

    var_sentence = ""
    if cv_a is not None:
        if cv_a <= 0.06:
            var_sentence = f"Inter-annual variability is <b>low</b> (CV {_fmt_pct_from_frac(cv_a)}), suggesting a comparatively stable yield profile."
        elif cv_a <= 0.12:
            var_sentence = f"Inter-annual variability is <b>moderate</b> (CV {_fmt_pct_from_frac(cv_a)}), which is typical for many sites."
        else:
            var_sentence = f"Inter-annual variability is <b>elevated</b> (CV {_fmt_pct_from_frac(cv_a)}), indicating higher production uncertainty year-to-year."

    # ------------------------------
    # 2) Economic attractiveness statement
    # ------------------------------
    npv_p50 = _safe_float(npv.get("p50"))
    npv_p10 = _safe_float(npv.get("p10"))
    npv_p90 = _safe_float(npv.get("p90"))

    irr_p50 = _safe_float(irr.get("p50"))
    irr_p10 = _safe_float(irr.get("p10"))
    irr_p90 = _safe_float(irr.get("p90"))

    # Build a screening-level statement from available NPV/IRR values.
    if npv_p50 is None and irr_p50 is None:
        econ_sentence = "The payload does not contain sufficient economic KPIs (NPV/IRR) to form an investment-style screening statement."
    else:
        if npv_p50 is not None:
            if npv_p50 >= 1_000_000:
                econ_sentence = f"Economically, the project screens <b>attractive</b> with a p50 NPV of <b>{_fmt_eur(npv_p50)}</b>."
            elif npv_p50 >= 0:
                econ_sentence = f"Economically, the project screens <b>borderline-positive</b> with a p50 NPV of <b>{_fmt_eur(npv_p50)}</b>."
            else:
                econ_sentence = f"Economically, the project screens <b>challenging</b> with a p50 NPV of <b>{_fmt_eur(npv_p50)}</b>."
        else:
            econ_sentence = "Economically, NPV is not provided at p50 level in the payload."

        # Where WACC is available, compare IRR to WACC to express spread/buffer.
        if irr_p50 is not None and wacc is not None:
            margin = irr_p50 - wacc
            if margin >= 0.03:
                econ_sentence += f" The p50 IRR of <b>{_fmt_pct_from_frac(irr_p50)}</b> provides a <b>comfortable spread</b> over the WACC ({_fmt_pct_from_frac(wacc)})."
            elif margin >= 0.0:
                econ_sentence += f" The p50 IRR of <b>{_fmt_pct_from_frac(irr_p50)}</b> is <b>only slightly above</b> the WACC ({_fmt_pct_from_frac(wacc)}), implying limited buffer."
            else:
                econ_sentence += f" The p50 IRR of <b>{_fmt_pct_from_frac(irr_p50)}</b> is <b>below</b> the WACC ({_fmt_pct_from_frac(wacc)}), which is typically not investment-grade without improvements."
        elif irr_p50 is not None and wacc is None:
            econ_sentence += f" The p50 IRR is <b>{_fmt_pct_from_frac(irr_p50)}</b>; however, a WACC/discount rate is not available for a spread assessment."

    # Add uncertainty range sentence if available.
    uncert_sentence = ""
    if (npv_p10 is not None or npv_p90 is not None) and (irr_p10 is not None or irr_p90 is not None):
        uncert_sentence = (
            f"Uncertainty bands in the payload indicate NPV p10/p90 of <b>{_fmt_eur(npv_p10)}</b> / <b>{_fmt_eur(npv_p90)}</b> "
            f"and IRR p10/p90 of <b>{_fmt_pct_from_frac(irr_p10)}</b> / <b>{_fmt_pct_from_frac(irr_p90)}</b>."
        )
    elif npv_p10 is not None or npv_p90 is not None:
        uncert_sentence = f"Uncertainty bands show NPV p10/p90 of <b>{_fmt_eur(npv_p10)}</b> / <b>{_fmt_eur(npv_p90)}</b>."
    elif irr_p10 is not None or irr_p90 is not None:
        uncert_sentence = f"Uncertainty bands show IRR p10/p90 of <b>{_fmt_pct_from_frac(irr_p10)}</b> / <b>{_fmt_pct_from_frac(irr_p90)}</b>."

    # ------------------------------
    # 3) Risk indicator sentence (probabilities)
    # ------------------------------
    risk_sentence = ""
    if prob_npv_lt0 is not None:
        p = _safe_float(prob_npv_lt0)
        if p is not None:
            if p <= 0.20:
                risk_sentence = f"Downside risk appears <b>low</b> with Prob(NPV&lt;0) of <b>{_fmt_pct_from_frac(p)}</b>."
            elif p <= 0.40:
                risk_sentence = f"Downside risk appears <b>moderate</b> with Prob(NPV&lt;0) of <b>{_fmt_pct_from_frac(p)}</b>."
            else:
                risk_sentence = f"Downside risk appears <b>material</b> with Prob(NPV&lt;0) of <b>{_fmt_pct_from_frac(p)}</b>."

    # If NPV downside probability isn't available, fall back to an IRR-vs-WACC probability if present.
    if prob_irr_lt_wacc is not None and (risk_sentence == ""):
        p2 = _safe_float(prob_irr_lt_wacc)
        if p2 is not None:
            risk_sentence = f"The payload indicates Prob(IRR&lt;WACC) of <b>{_fmt_pct_from_frac(p2)}</b>."

    # ------------------------------
    # Compose final text
    # ------------------------------
    if lat is not None and lon is not None:
        loc_str = f"Selected location: <b>{lat}, {lon}</b>. "
    else:
        loc_str = "Selected location coordinates are not fully available in the payload. "

    parts = [loc_str, cf_sentence, aep_sentence]
    if var_sentence:
        parts.append(var_sentence)
    parts.append(econ_sentence)
    if uncert_sentence:
        parts.append(uncert_sentence)
    if risk_sentence:
        parts.append(risk_sentence)

    # Remove empty segments and join into a single paragraph.
    return " ".join([p.strip() for p in parts if p and p.strip()])


# ============================
# PDF builder (4 pages)
# ============================

# Static introduction text displayed on the PDF's first page.
intro_text = (
    "This report provides an automated, site-specific assessment of expected wind energy production "
    "and project economics based on the inputs selected on the website. It combines a long-term "
    "energy forecast at monthly resolution with a financial model that translates production into "
    "revenues, costs, and value metrics such as NPV and IRR. All figures and tables are indicative and intended for "
    "early-stage screening; final investment decisions require detailed engineering, permitting, "
    "and commercial due diligence."
)


def build_report_pdf_bytes(*, project_meta: dict, payload: dict, logo_path: Path = LOGO_PATH) -> bytes:
    """
    Generate a multi-page PDF report as bytes.

    Layout fixes:
      - Smaller charts (page 2/3/4) and less "after chart" whitespace
      - OPEX table no longer squeezed into footer
      - Decision summary no longer collides with disclaimer/footer
      - Annual energy year axis uses integer ticks
      - CAPEX chart uses labels (not "category index")
    """

    # ---------- chart sizing knobs ----------
    # Centralized chart heights simplify layout adjustments without needing to modify each call-site.
    H_ENERGY = 300   # page 2 chart height (was 380+)
    H_PRICE  = 200   # page 3
    H_CAPEX  = 190   # page 3
    H_CUMCF  = 190   # page 4
    H_TORN   = 190   # page 4
    CHART_GAP_AFTER = 15  # reduce from ~18 (less bottom waste)

    # ---------- helper: chart -> PNG bytes ----------
    # Charts are rendered to PNG and embedded as images into the PDF.
    # This avoids font/rendering inconsistencies across platforms and keeps ReportLab layout deterministic.
    def fig_to_png_bytes(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ---------- parse rated power ----------
    # Rated power is used for capacity factor computation. Parsing is intentionally tolerant.
    rated_power_mw = project_meta.get("rated_power_mw", None)
    try:
        rated_power_mw = float(rated_power_mw) if rated_power_mw is not None else None
    except Exception:
        rated_power_mw = None

    # ---------- normalize energy inputs ----------
    # The payload can include:
    # - energy_forecast: raw forecast series (timestamp, energy_kwh)
    # - mwh_monthly_used: normalized monthly series already prepared by /api/calc
    energy_forecast = payload.get("energy_forecast")
    monthly_mwh = payload.get("mwh_monthly_used")

    # NEW: prefer yearly series already computed in /api/calc
    annual_mwh_series: List[Dict[str, Any]] = []
    yearly_used = payload.get("mwh_yearly_used") or []

    if yearly_used:
        # /api/calc format:
        # [{"calendar_year": 2026, "mwh": 12345.6}, ...]
        for r in yearly_used:
            try:
                y = int(r.get("calendar_year"))
                v = float(r.get("mwh"))
            except Exception:
                continue
            annual_mwh_series.append({"year": y, "mwh": v})

        # sort by year just in case
        annual_mwh_series.sort(key=lambda x: x["year"])

    else:
        # fallback 1: derive from energy_forecast if monthly not present
        if (not monthly_mwh) and energy_forecast:
            monthly_mwh, annual_mwh_series = derive_monthly_and_annual_energy(energy_forecast)
            payload["mwh_monthly_used"] = monthly_mwh

        # fallback 2: derive yearly from monthly_mwh if present
        elif monthly_mwh:
            by_year: Dict[int, float] = {}
            for r in monthly_mwh:
                ms = str(r.get("month_start", ""))
                if len(ms) >= 7 and ms[:4].isdigit():
                    y = int(ms[:4])
                    try:
                        by_year[y] = by_year.get(y, 0.0) + float(r.get("mwh"))
                    except Exception:
                        pass
            annual_mwh_series = [{"year": y, "mwh": by_year[y]} for y in sorted(by_year.keys())]

    # Debug prints aid operational troubleshooting when PDFs are generated server-side.
    print("[PDF] annual_mwh_series len:", len(annual_mwh_series))
    print("[PDF] annual_mwh_series first rows:", annual_mwh_series[:3])

    # Compute KPIs and attach to payload for downstream table rendering.
    if monthly_mwh and rated_power_mw:
        aep, cf = derive_aep_and_capacity_factor(monthly_mwh, rated_power_mw)
        payload.setdefault("metrics", {})
        payload["metrics"].setdefault("aep_mwh_per_year", aep)
        payload["metrics"].setdefault("capacity_factor_pct", cf)

    def _derive_seasonality_and_variability(monthly_mwh: List[Dict[str, Any]]):
        """
        Derive:
          - seasonality profile: average MWh for each calendar month across available years
          - annual variability: min/mean/max and coefficient of variation (CV%)

        Inputs expect month_start formatted as "YYYY-MM-01" (or compatible).
        """
        # Month buckets for seasonality (1..12).
        by_month = {m: [] for m in range(1, 13)}
        # Annual sums per year.
        by_year = {}

        for r in monthly_mwh or []:
            ms = str(r.get("month_start", ""))
            if len(ms) < 7 or not ms[:4].isdigit():
                continue
            y = int(ms[:4])
            m = int(ms[5:7]) if ms[5:7].isdigit() else None
            try:
                v = float(r.get("mwh"))
            except Exception:
                continue

            if m in by_month:
                by_month[m].append(v)
            by_year[y] = by_year.get(y, 0.0) + v

        # seasonality profile: avg MWh per month
        seasonality = [sum(by_month[m]) / len(by_month[m]) if by_month[m] else None for m in range(1, 13)]

        # annual variability: min/mean/max and coeff. of variation (CV)
        annual_vals = list(by_year.values())
        if annual_vals:
            mean_a = sum(annual_vals) / len(annual_vals)
            min_a = min(annual_vals)
            max_a = max(annual_vals)
            # CV (%): std/mean
            var = sum((x - mean_a) ** 2 for x in annual_vals) / max(1, (len(annual_vals) - 1))
            std = var ** 0.5
            cv_pct = (std / mean_a) * 100.0 if mean_a else None
        else:
            mean_a = min_a = max_a = cv_pct = None

        return seasonality, min_a, mean_a, max_a, cv_pct

    # ============================
    # Charts
    # ============================

    # Figure 1: Annual energy (make matplotlib itself compact)
    # Note: chart1_png is currently not embedded later; retained for potential future layout changes.
    chart1_png = None
    if annual_mwh_series:
        years, vals = [], []
        for r in annual_mwh_series:
            try:
                years.append(int(r["year"]))
                vals.append(float(r.get("mwh") or 0.0))
            except Exception:
                continue

        if years and vals:
            fig1 = plt.figure(figsize=(5.2, 2.6))  # smaller plot
            ax = fig1.add_subplot(111)
            ax.plot(years, vals, marker="o", linewidth=1.2, markersize=3.5)
            ax.set_title("Annual Energy Production", fontsize=10)
            ax.set_xlabel("Year", fontsize=9)
            ax.set_ylabel("Energy (MWh)", fontsize=9)
            # Ensure year ticks are integers and limited to a readable count.
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            ax.tick_params(axis="both", labelsize=8)
            chart1_png = fig_to_png_bytes(fig1)

    # Seasonality and variability are derived from monthly_mwh and used on the energy page.
    seasonality, min_a, mean_a, max_a, cv_pct = _derive_seasonality_and_variability(monthly_mwh)

    chart_season_png = None
    if seasonality and any(v is not None for v in seasonality):
        # Replace Nones with 0 for plotting to preserve month positions.
        ys = [0.0 if v is None else float(v) for v in seasonality]
        figS = plt.figure(figsize=(5.2, 1.6))
        ax = figS.add_subplot(111)
        ax.bar(range(1, 13), ys)
        ax.set_title("Seasonality (Average Monthly Energy)", fontsize=9)
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel("MWh", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        chart_season_png = fig_to_png_bytes(figS)

    # Figure 2 (REPLACEMENT): Annual Revenue quantiles from yearly_table
    # The yearly_table aggregates per project year and already includes revenue quantiles.
    yearly_tbl = payload.get("yearly_table") or []
    rev_x, rev_p10, rev_p50, rev_p90 = [], [], [], []

    for r in yearly_tbl:
        try:
            y = int(r.get("year"))
        except Exception:
            continue

        # Local helper to safely extract numeric series columns.
        def _f(k):
            try:
                v = r.get(k)
                return float(v) if v is not None else None
            except Exception:
                return None

        p10 = _f("revenue_p10")
        p50 = _f("revenue_p50")
        p90 = _f("revenue_p90")

        if p10 is None and p50 is None and p90 is None:
            continue

        rev_x.append(y)
        rev_p10.append(p10)
        rev_p50.append(p50)
        rev_p90.append(p90)

    chart2_png = None
    if rev_x:
        fig2 = plt.figure(figsize=(5.6, 2.6))
        ax = fig2.add_subplot(111)

        # Plot each quantile only if any values exist to avoid empty legend entries.
        if any(v is not None for v in rev_p10):
            ax.plot(rev_x, [0.0 if v is None else v for v in rev_p10], label="Revenue P10", linewidth=1.2)
        if any(v is not None for v in rev_p50):
            ax.plot(rev_x, [0.0 if v is None else v for v in rev_p50], label="Revenue P50", linewidth=1.2)
        if any(v is not None for v in rev_p90):
            ax.plot(rev_x, [0.0 if v is None else v for v in rev_p90], label="Revenue P90", linewidth=1.2)

        ax.set_title("Annual Revenue (P10 / P50 / P90)", fontsize=10)
        ax.set_xlabel("Project year", fontsize=9)
        ax.set_ylabel("Revenue (EUR)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(fontsize=8, frameon=False)

        chart2_png = fig_to_png_bytes(fig2)

    # Figure 3 (REPLACEMENT): CAPEX vs OPEX totals (since breakdown not provided)
    # Aggregated totals provide a compact "cost structure" snapshot even without component breakdown.
    capex_total = payload.get("capex_total_eur", None)
    opex_total = payload.get("opex_total_eur", None)

    def _to_float(x):
        # Local float conversion helper for chart preparation.
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    capex_total_f = _to_float(capex_total)
    opex_total_f = _to_float(opex_total)

    chart3_png = None
    if (capex_total_f is not None) or (opex_total_f is not None):
        labels = ["CAPEX total", "OPEX total (20y)"]
        values = [
            0.0 if capex_total_f is None else capex_total_f,
            0.0 if opex_total_f is None else opex_total_f,
        ]

        fig3 = plt.figure(figsize=(5.6, 2.6))
        ax = fig3.add_subplot(111)
        ax.bar(range(len(values)), values)
        ax.set_title("Total Costs (CAPEX vs OPEX)", fontsize=10)
        ax.set_ylabel("EUR", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

        chart3_png = fig_to_png_bytes(fig3)

    # Figure 4 (REPLACEMENT): Cumulative Profit (P50) from yearly_table
    # Uses undiscounted cumulative profit to visualize payback-like dynamics.
    profit_x, profit_p50 = [], []
    cum = 0.0
    have_any = False

    for r in (payload.get("yearly_table") or []):
        try:
            y = int(r.get("year"))
        except Exception:
            continue
        try:
            p50 = r.get("profit_p50")
            p50 = float(p50) if p50 is not None else None
        except Exception:
            p50 = None

        if p50 is None:
            continue

        have_any = True
        cum += p50
        profit_x.append(y)
        profit_p50.append(cum)

    chart4_png = None
    if have_any:
        fig4 = plt.figure(figsize=(5.6, 2.6))
        ax = fig4.add_subplot(111)
        ax.plot(profit_x, profit_p50, linewidth=1.2)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title("Cumulative Profit (P50) – Not Discounted", fontsize=10)
        ax.set_xlabel("Project year", fontsize=9)
        ax.set_ylabel("Cumulative profit (EUR)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        chart4_png = fig_to_png_bytes(fig4)

    # Figure 5 (REPLACEMENT): NPV uncertainty range from quantiles
    # Visualizes spread across P10 / mean / P90 for a quick risk snapshot.
    npv_q = payload.get("npv_eur", {}) or {}
    try:
        npv_p10 = float(npv_q.get("p10")) if npv_q.get("p10") is not None else None
    except Exception:
        npv_p10 = None
    try:
        npv_mean = float(npv_q.get("mean")) if npv_q.get("mean") is not None else None
    except Exception:
        npv_mean = None
    try:
        npv_p90 = float(npv_q.get("p90")) if npv_q.get("p90") is not None else None
    except Exception:
        npv_p90 = None

    chart5_png = None
    if (npv_p10 is not None) or (npv_mean is not None) or (npv_p90 is not None):
        fig5 = plt.figure(figsize=(5.6, 2.6))
        ax = fig5.add_subplot(111)

        labels = ["NPV P10", "NPV Mean", "NPV P90"]
        vals = [
            0.0 if npv_p10 is None else npv_p10,
            0.0 if npv_mean is None else npv_mean,
            0.0 if npv_p90 is None else npv_p90,
        ]
        ax.bar(range(3), vals)
        ax.set_title("NPV Uncertainty (P10 / Mean / P90)", fontsize=10)
        ax.set_ylabel("EUR", fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

        chart5_png = fig_to_png_bytes(fig5)

    # ============================
    # PDF composition
    # ============================
    # ReportLab style setup for consistent typography across paragraphs.
    styles = getSampleStyleSheet()
    style_body = styles["BodyText"]
    style_body.fontName = "Helvetica"
    style_body.fontSize = 10
    style_body.leading = 13

    # Memory buffer used to collect the final PDF bytes.
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Layout constants and margins.
    MARGIN_L = 40
    MARGIN_R = 40
    TOP = height - 40
    FOOT = 35

    def draw_header(page_no: int):
        """
        Draw a consistent header on each page:
          - Title text on the left
          - Optional logo on the right
          - Separator line below
        """
        c.setFont("Helvetica", 10)

        # Left: report title
        c.drawString(
            MARGIN_L,
            TOP,
            "Wind Site Assessment – Summary Report"
        )

        # Right: logo
        LOGO_WIDTH = 90    # points
        LOGO_HEIGHT = 32   # points

        if logo_path and Path(logo_path).exists():
            try:
                logo = ImageReader(str(logo_path))
                c.drawImage(
                    logo,
                    width - MARGIN_R - LOGO_WIDTH,
                    TOP - LOGO_HEIGHT + 4,
                    width=LOGO_WIDTH,
                    height=LOGO_HEIGHT,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception as e:
                # Logo failures should not fail the report; degrade gracefully.
                print("[PDF] Failed to load header logo:", e)

        # Separator line
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(1)
        c.line(MARGIN_L, TOP - 14, width - MARGIN_R, TOP - 14)


    def draw_footer(page_no: int):
        """
        Draw a consistent footer:
          - Separator line above footer area
          - Bracket-value note (templated field convention)
          - Right-aligned page numbering
        """
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(1)
        c.line(MARGIN_L, FOOT + 18, width - MARGIN_R, FOOT + 18)
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(MARGIN_L, FOOT, "All bracketed values [LIKE_THIS] are populated by the application.")
        c.drawRightString(width - MARGIN_R, FOOT, f"Page {page_no}")
        c.setFillColor(colors.black)

    def draw_paragraph(text: str, x: float, y: float, w: float) -> float:
        """
        Draw a ReportLab Paragraph at (x, y) with available width w.
        Returns the new y position after the paragraph is rendered (top-down flow).
        """
        p = Paragraph(text, style_body)
        _, ph = p.wrap(w, 1000)
        p.drawOn(c, x, y - ph)
        return y - ph

    def draw_table(
        data: List[List[str]],
        x: float,
        y: float,
        col_widths: List[float],
        font_size: int = 10,
        header_font_size: int = 10,
    ) -> float:
        """
        Draw a table with a simple, consistent style (grid + header shading).
        Returns the new y position after rendering.
        """
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", header_font_size),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        tw, th = t.wrapOn(c, sum(col_widths), 1000)
        t.drawOn(c, x, y - th)
        return y - th

    def draw_chart(title: str, png_bytes: Optional[bytes], x: float, y: float, w: float, h: float) -> float:
        """
        Draw a chart frame with title and embedded PNG image.
        If png_bytes is missing, a placeholder message is displayed.
        Returns the new y position after rendering (including chart gap).
        """
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, title)
        y -= 12

        # Draw bounding box to keep layout consistent even when image is missing.
        c.setStrokeColor(colors.HexColor("#999999"))
        c.rect(x, y - h, w, h, stroke=1, fill=0)

        if png_bytes:
            img = ImageReader(io.BytesIO(png_bytes))
            c.drawImage(
                img,
                x + 6,
                y - h + 6,
                width=w - 12,
                height=h - 12,
                preserveAspectRatio=True,
                mask="auto"
            )
        else:
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.grey)
            c.drawString(x + 12, y - 22, "CHART PLACEHOLDER - Data not available")
            c.setFillColor(colors.black)

        return y - h - CHART_GAP_AFTER

    # ============================
    # Page 1: Summary + Executive assessment (NEW)
    # ============================
    draw_header(1)

    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN_L, height - 130, "Wind Site Assessment & Financial Summary")
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN_L, height - 155, "Automated report generated from website outputs (4 pages).")

    # Intro paragraph below title
    intro_y = height - 175
    intro_y = draw_paragraph(intro_text, MARGIN_L, intro_y, width - MARGIN_L - MARGIN_R)

    # --- NEW: executive location assessment right after intro ---
    # Park size is required for interpreting energy and capacity factor in absolute terms.
    park_mw = payload.get("park_mw") or project_meta.get("rated_power_mw") or 0.0
    try:
        park_mw = float(park_mw) if park_mw is not None else 0.0
    except Exception:
        park_mw = 0.0

    # monthly_mwh: [{"month_start":"YYYY-MM-DD","mwh":...}, ...]
    # Convert to tuples for helper functions that use datetime keys.
    monthly_tuples: List[Tuple[datetime, float]] = []
    for r in (monthly_mwh or []):
        ms = str(r.get("month_start", ""))  # "YYYY-MM-01" or "YYYY-MM-DD"
        try:
            dt = datetime.fromisoformat(ms)
            v = float(r.get("mwh"))
            monthly_tuples.append((dt, v))
        except Exception:
            continue

    # annual_mwh_series: [{"year": YYYY, "mwh": ...}, ...]
    annual_tuples: List[Tuple[int, float]] = []
    for r in (annual_mwh_series or []):
        try:
            annual_tuples.append((int(r.get("year")), float(r.get("mwh"))))
        except Exception:
            continue

    assessment_html = _location_assessment_text(
        payload=payload,
        monthly=monthly_tuples,
        annual=annual_tuples,
        park_mw=park_mw,
    )

    intro_y -= 10
    intro_y = draw_paragraph(
        f"<b>Executive location assessment:</b> {assessment_html}",
        MARGIN_L,
        intro_y,
        width - MARGIN_L - MARGIN_R,
    )

    # Start the two-column tables below the intro + assessment
    y_top = intro_y - 14

    gutter = 18
    col_w = (width - MARGIN_L - MARGIN_R - gutter) / 2
    x_left = MARGIN_L
    x_right = MARGIN_L + col_w + gutter

    # Extract commonly displayed inputs with placeholders for templating and missing values.
    equity_share = payload.get("equity_share", None)
    eeg_support = payload.get("eeg_on", None)
    eeg_uplift_total = payload.get("eeg_uplift_total_eur", None)
    lat = _val(project_meta, "lat", _val(payload, "latitude", "[LAT]"))
    lon = _val(project_meta, "lon", _val(payload, "longitude", "[LON]"))

    turbine_model = payload.get("turbine_model") or _val(project_meta, "turbine_model", "[TURBINE_MODEL]")
    rated_power_txt = _val(project_meta, "rated_power_mw", "[RATED_POWER_MW]")
    hub_height = _val(project_meta, "hub_height_m", "[HUB_HEIGHT_M]")

    # Determine project horizon years primarily from meta; otherwise infer from COD.
    cod = project_meta.get("cod_date") or payload.get("cod_date")
    cod_s = str(cod) if cod is not None else ""

    start_year = project_meta.get("start_year")
    end_year = project_meta.get("end_year")

    if not start_year:
        start_year = cod_s[:4] if len(cod_s) >= 4 and cod_s[:4].isdigit() else "[START_YEAR]"
    if not end_year:
        end_year = str(int(start_year) + 20) if str(start_year).isdigit() else "[END_YEAR]"

    report_date = _val(project_meta, "report_date", _today_str())

    left_table = [
        ["Site & Financing Inputs", "Value"],
        ["Coordinates", f"{lat}, {lon}"],
        ["Turbine model", str(turbine_model)],
        ["Rated power", f"{rated_power_txt} MW"],
        ["Hub height", f"{hub_height} m"],
        ["Horizon", f"{start_year} - {end_year}"],
        ["Equity share", _fmt_pct(equity_share, "[EQUITY_SHARE_PCT]")],
        ["EEG support applied", _fmt_yesno(eeg_support, "[EEG_SUPPORT]")],
        ["EEG uplift total (EUR)", _fmt_num(eeg_uplift_total, "[EEG_UPLIFT_TOTAL_EUR]")],
        ["Report date", str(report_date)],
    ]
    y_left_end = draw_table(left_table, x_left, y_top, [col_w * 0.55, col_w * 0.45])

    # --- Scenario Snapshot derived from payload quantiles ---
    # Uses quantiles from Monte Carlo summary for a compact best/base/worst view.
    npv_q = payload.get("npv_eur", {}) or {}
    irr_q = payload.get("irr", {}) or {}

    def _irr_to_pct(x):
        # IRR is expected as a fraction (e.g., 0.12). Convert to percent for display.
        try:
            return float(x) * 100.0 if x is not None else None
        except Exception:
            return None

    best_npv = npv_q.get("p90", None)
    base_npv = npv_q.get("mean", None)
    worst_npv = npv_q.get("p10", None)

    best_irr = _irr_to_pct(irr_q.get("p90", None))
    base_irr = _irr_to_pct(irr_q.get("mean", None))
    worst_irr = _irr_to_pct(irr_q.get("p10", None))

    right_table = [
        ["Scenario Snapshot (from quantiles)", ""],
        ["Scenario", "NPV / IRR"],
        ["Best (P90)", f"{_fmt_num(best_npv,'[NPV_P90]')} / {_fmt_pct(best_irr,'[IRR_P90_PCT]')}"],
        ["Base (Mean)", f"{_fmt_num(base_npv,'[NPV_MEAN]')} / {_fmt_pct(base_irr,'[IRR_MEAN_PCT]')}"],
        ["Worst (P10)", f"{_fmt_num(worst_npv,'[NPV_P10]')} / {_fmt_pct(worst_irr,'[IRR_P10_PCT]')}"],
    ]

    y_right_end = draw_table(right_table, x_right, y_top, [col_w * 0.35, col_w * 0.65])

    # Continue below whichever column ended lower to avoid overlap.
    y = min(y_left_end, y_right_end) - 16

    metrics = payload.get("metrics", {}) or {}

    aep = _val(metrics, "aep_mwh_per_year", "[AEP_MWH_PER_YEAR]")
    cf = _val(metrics, "capacity_factor_pct", "[CAPACITY_FACTOR_PCT]")
    payback = int(end_year) - int(start_year) if str(start_year).isdigit() and str(end_year).isdigit() else 20

    # --- NPV/IRR: use MEAN from payload summaries (not p50) ---
    # Mean is used here as the "base case" convention for this report layout.
    npv_mean = payload.get("npv_eur", {}).get("mean", None)

    irr_mean_frac = payload.get("irr", {}).get("mean", None)
    try:
        irr_mean_pct = float(irr_mean_frac) * 100.0 if irr_mean_frac is not None else None
    except Exception:
        irr_mean_pct = None

    npv_base = npv_mean if npv_mean is not None else "[NPV_MEAN]"
    irr_base = irr_mean_pct if irr_mean_pct is not None else "[IRR_MEAN_PCT]"

    key_table = [
        ["Key Results (Base Case)", "Value"],
        ["AEP (annual MWh)", _fmt_num(aep, "[AEP_MWH_PER_YEAR]")],
        ["Capacity factor", _fmt_pct(cf, "[CAPACITY_FACTOR_PCT]")],
        ["NPV (at WACC)", _fmt_num(npv_base, "[NPV_BASE]")],
        ["IRR", _fmt_pct(irr_base, "[IRR_BASE_PCT]")],
        ["Discounted payback", f"{_fmt_num(payback,'[PAYBACK_YEARS]')} years"],
    ]
    draw_table(key_table, MARGIN_L, y, [240, width - MARGIN_L - MARGIN_R - 240])

    draw_footer(1)
    c.showPage()

    # ============================
    # Page 2: Energy Yield (smaller chart)
    # ============================
    draw_header(2)
    y = height - 80

    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN_L, y, "1. Energy Yield")
    y -= 18

    y = draw_paragraph(
        "Annual energy is computed by aggregating monthly forecasted energy values. "
        "This report presents a single production path.",
        MARGIN_L, y, width - MARGIN_L - MARGIN_R
    )
    y -= 8

    # NOTE: The title states revenue; chart2_png is a revenue chart derived from yearly_table.
    # If the energy page should show energy instead, chart1_png/annual energy can be wired here.
    y = draw_chart(
        "Figure 2 - Annual Revenue (P10/P50/P90)",
        chart2_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_PRICE
    )

    # KPI strip is a compact 1-row table for quick scanning.
    kpi_strip = [
        ["AEP (MWh/yr)", "Capacity factor", "Rated power", "Horizon"],
        [_fmt_num(aep, "[AEP_MWH_PER_YEAR]"),
         _fmt_pct(cf, "[CAPACITY_FACTOR_PCT]"),
         f"{_val(project_meta,'rated_power_mw','[RATED_POWER_MW]')} MW",
         f"{_val(project_meta,'start_year','[START_YEAR]')} - {_val(project_meta,'end_year','[END_YEAR]')}"],
    ]
    y = draw_table(
        kpi_strip,
        MARGIN_L,
        y,
        [(width - MARGIN_L - MARGIN_R) / 4] * 4,
        font_size=9,
        header_font_size=9
    )

    y -= 14
    gutter = 18
    col_w = (width - MARGIN_L - MARGIN_R - gutter) / 2
    x_left = MARGIN_L
    x_right = MARGIN_L + col_w + gutter

    # Left column: monthly seasonality bar chart.
    y_left = draw_chart(
        "Seasonality - Average Monthly Energy",
        chart_season_png,
        x_left,
        y,
        col_w,
        140
    )

    # Right column: annual variability summary table.
    var_table = [
        ["Variability (Annual Energy)", "Value"],
        ["Min / Mean / Max (MWh)",
         f"{_fmt_num(min_a,'[MIN_A]')} / {_fmt_num(mean_a,'[MEAN_A]')} / {_fmt_num(max_a,'[MAX_A]')}"],
        ["Inter-annual variability (CV)", _fmt_pct(cv_pct, "[CV_PCT]")],
    ]
    y_right = draw_table(
        var_table,
        x_right,
        y,
        [col_w * 0.55, col_w * 0.45],
        font_size=9,
        header_font_size=9
    )

    # Continue below the lower of the two columns.
    y = min(y_left, y_right)

    draw_footer(2)
    c.showPage()

    # ============================
    # Page 3: Price and Costs (avoid bottom squeeze)
    # ============================
    draw_header(3)
    y = height - 80

    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN_L, y, "2. Price and Costs")
    y -= 18

    y = draw_paragraph(
        "Revenue is calculated as energy multiplied by the applicable power price. "
        "If EEG support is enabled, the EEG premium is applied according to the configured scheme. "
        "Costs are separated into one-time CAPEX and recurring OPEX.",
        MARGIN_L, y, width - MARGIN_L - MARGIN_R
    )
    y -= 8

    # Power price path chart placeholder currently uses chart2_png (revenue quantiles).
    # If a distinct price path series becomes available, this can be replaced without layout changes.
    y = draw_chart(
        "Figure 2 - Power Price Path (Best/Base/Worst)",
        chart2_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_PRICE
    )

    # CAPEX vs OPEX total chart.
    y = draw_chart(
        "Figure 3 - Total Costs (CAPEX vs OPEX)",
        chart3_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_CAPEX
    )

    y -= 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN_L, y, "OPEX Summary (Base Case)")
    y -= 10

    yearly_tbl = payload.get("yearly_table") or []

    # Extract OPEX time series for summary stats.
    opex_years = []
    for r in yearly_tbl:
        try:
            v = r.get("opex")
            if v is None:
                continue
            opex_years.append(float(v))
        except Exception:
            continue

    opex_first_year = opex_years[0] if opex_years else None
    opex_avg_year = (sum(opex_years) / len(opex_years)) if opex_years else None
    opex_total = payload.get("opex_total_eur", None)

    opex_table = [
        ["Item", "Value"],
        ["OPEX (year 1)", _fmt_num(opex_first_year, "[OPEX_Y1]")],
        ["Avg. annual OPEX", _fmt_num(opex_avg_year, "[OPEX_AVG]")],
        ["Total OPEX (20y)", _fmt_num(opex_total, "[OPEX_TOTAL]")],
        ["Notes", "Detailed OPEX components not provided by model output."],
    ]
    draw_table(
        opex_table,
        MARGIN_L,
        y,
        [200, width - MARGIN_L - MARGIN_R - 200],
        font_size=9,
        header_font_size=9
    )

    draw_footer(3)
    c.showPage()

    # ============================
    # Page 4: Financial Results and Risk (avoid bottom squeeze)
    # ============================
    draw_header(4)
    y = height - 80

    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN_L, y, "3. Financial Results and Risk")
    y -= 18

    y = draw_paragraph(
        "Cashflows are discounted using WACC. Financing inputs (equity share and debt share) "
        "and policy support settings (EEG) are reflected in the economic model configuration.",
        MARGIN_L, y, width - MARGIN_L - MARGIN_R
    )
    y -= 8

    # Cumulative profit chart (undiscounted) for an intuitive trajectory.
    y = draw_chart(
        "Figure 4 - Cumulative Profit (P50, not discounted)",
        chart4_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_CUMCF
    )

    # NPV uncertainty chart highlighting downside vs upside spread.
    y = draw_chart(
        "Figure 5 - NPV Uncertainty (P10/Mean/P90)",
        chart5_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_TORN
    )

    y -= 6
    # Place disclaimer above footer with a minimum y to prevent overlap in tight layouts.
    draw_paragraph(
        "<font size='9' color='#666666'>Disclaimer: This report is an indicative, model-based assessment generated automatically. "
        "It does not constitute investment, legal, or engineering advice.</font>",
        MARGIN_L, max(y, FOOT + 70), width - MARGIN_L - MARGIN_R
    )

    draw_footer(4)
    c.showPage()
    c.save()

    buf.seek(0)
    return buf.read()


def send_calc_email(
    to_email: str,
    *,
    project_meta: dict,
    result_meta: dict,
    pdf_bytes: bytes | None = None,
    pdf_filename: str = "wind_report.pdf",
) -> None:
    """
    Send an email containing calculation results, optionally including the generated PDF report.

    Email provider:
      - Maileroo API (SMTP API)
    Configuration:
      - API_KEY: Maileroo API key
      - FROM_ADDR: verified sender address
    """
    api_key = os.getenv("API_KEY")
    from_addr = os.getenv("FROM_ADDR")
    from_name = "Wind Model Backend Team"

    # If email configuration is missing, skip sending rather than failing the request.
    if not api_key or not from_addr:
        print("[email] MAILEROO_API_KEY / MAILEROO_FROM not set. Skipping email.")
        return

    subject = "Your wind park analysis is ready"

    # Plain-text email body for broad compatibility.
    body = (
        "Hi there,\n\n"
        "Thanks for trying our wind park analysis tool.\n\n"
        "Your results are ready — we’ve attached a PDF report with the key metrics and insights for your project.\n\n"
        "You can also jump back into the app anytime to explore the details.\n\n"
        "Questions or feedback? Just reply to this email — we’re happy to help.\n\n"
        "Best,\n"
        "The Wind Side Analytics Team\n"
    )

    payload = {
        "from": {"address": from_addr, "display_name": from_name},
        "to": {"address": to_email},
        "subject": subject,
        "plain": body,
    }

    # Attach PDF (optional)
    # Maileroo expects base64-encoded attachment content.
    if pdf_bytes:
        payload["attachments"] = [
            {
                "file_name": pdf_filename,
                "content_type": "application/pdf",
                "content": base64.b64encode(pdf_bytes).decode("ascii"),
                "inline": False,
            }
        ]

    try:
        resp = requests.post(
            "https://smtp.maileroo.com/api/v2/emails",
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": api_key,
            },
            json=payload,
            timeout=20,
        )

        # Treat any 4xx/5xx as an error; log body for provider diagnostics.
        if resp.status_code >= 400:
            print("[email] Maileroo error:", resp.status_code, resp.text)
            return

        print(f"[email] Sent calculation email to {to_email} via Maileroo.")
    except Exception as e:
        # Email failures should not crash the API request in most cases; log and continue.
        print(f"[email] Failed to send email to {to_email} via Maileroo: {e}")


# ============================================================
# JSON helpers (make pandas/numpy safe for JSON)
# ============================================================
def to_jsonable(obj: Any):
    """
    Convert pandas/numpy objects into JSON-serializable types and replace NaN/Inf with None.

    This is required because:
      - JSON does not support NaN/Infinity
      - pandas Timestamp and numpy scalar types are not directly serializable
    """
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        # Convert datetime columns to string to keep JSON safe and predictable.
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        # Replace infinities with NaN, then NaN with None.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        s = obj.copy()
        out = []
        for x in s.to_list():
            if pd.isna(x) or x in (np.inf, -np.inf):
                out.append(None)
            else:
                out.append(float(x))
        return out

    if isinstance(obj, pd.Timestamp):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return val

    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]

    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    return obj


def scrub_nonfinite(obj: Any):
    """
    Recursively replace NaN/Inf/-Inf with None in already JSON-like objects.

    This is applied as a last pass to guard against any non-finite values that
    may have been introduced after conversions or merges.
    """
    if isinstance(obj, dict):
        return {k: scrub_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_nonfinite(v) for v in obj]
    if isinstance(obj, np.generic):
        return scrub_nonfinite(obj.item())
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


# ============================================================
# Small helper: extract P10/P50/P90 from a "final_summary_df"
# ============================================================
def pick_summary_row(df: pd.DataFrame, metric_name: str) -> dict:
    """
    Extract mean and quantiles for a single metric from final_summary_df.

    final_summary_df format:
      metric | mean | p10 | p50 | p90

    Returns:
      {"mean": ..., "p10": ..., "p50": ..., "p90": ...}
    """
    if df is None or df.empty:
        return {}

    row = df.loc[df["metric"] == metric_name]
    if row.empty:
        return {}
    r = row.iloc[0].to_dict()
    return {"mean": r.get("mean"), "p10": r.get("p10"), "p50": r.get("p50"), "p90": r.get("p90")}


# ============================================================
# Forecast helpers
# ============================================================
def stable_seed(*, lat: float, lon: float, hub_height_m: float, turbine_type: str) -> int:
    """
    Deterministic seed derived from location and turbine parameters.

    Purpose:
      - Ensure repeatability for the same inputs (important for debugging and caching)
      - Provide stable randomness across deployments without storing state
    """
    s = f"{lat:.6f}|{lon:.6f}|{hub_height_m:.3f}|{turbine_type}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # Use a subset of the hash to fit within typical RNG seed ranges.
    return int(h[:8], 16)


def build_mwh_series_from_forecast(req: CalcRequest) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run the ML forecast pipeline and return:
      - monthly MWh pd.Series indexed by month-start (MS) covering FORECAST_MONTHS from COD
      - representative dataframe (rep) used to generate the monthly aggregation

    Operational notes:
      - fast_mode reduces ERA5 sampling fraction and number of MC simulations for responsiveness.
      - monthly series is forward-filled if the forecast produces gaps at the end of the horizon.
    """
    if req.latitude is None or req.longitude is None:
        raise ValueError("latitude/longitude required for forecast mode")

    # fast_mode is treated as a boolean, with None defaulting to False.
    forecast_fast = bool(req.fast_mode)

    # Normalize COD to month-start to align production aggregation and horizon indexing.
    cod = pd.Timestamp(req.cod_date)
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)
    horizon_months = int(FORECAST_MONTHS)
    target_index = pd.date_range(cod, periods=horizon_months, freq="MS")

    # Seed selection isolates randomness per scenario inputs.
    seed = stable_seed(
        lat=float(req.latitude),
        lon=float(req.longitude),
        hub_height_m=float(req.hub_height_m),
        turbine_type=req.turbine_type_id,
    )

    # Forecast performance parameters.
    # era5_frac < 1.0 subsamples historical weather for speed in fast mode.
    era5_frac = 0.10 if forecast_fast else 1.0
    n_sims = 150 if forecast_fast else 500
    era5_subsample_seed = 42
    mc_seed = seed

    t0 = perf_counter()
    hist = build_hist_monthly_energy_lookup(
        lat=float(req.latitude),
        lon=float(req.longitude),
        hub_height_m=float(req.hub_height_m),
        turbine_type_id=req.turbine_type_id,
        era5_main_dir=DEFAULT_ERA5_DIR,
        turbine_power_curve_json=DEFAULT_TURBINE_CURVES,
        model_dir=DEFAULT_MODEL_DIR,
        era5_sample_frac=era5_frac,
        random_seed=era5_subsample_seed,
    )
    rep = forecast_monthly_representative_path_from_hist_energy(
        hist_monthly=hist,
        start_date=req.cod_date,
        years=int(math.ceil(horizon_months / 12)),
        n_sims=n_sims,
        random_seed=mc_seed,
    )
    # Normalize timestamps to naive datetime (timezone removed) for consistent grouping/formatting.
    rep["timestamp"] = pd.to_datetime(rep["timestamp"], utc=True).dt.tz_convert(None)
    print("[FORECAST] ts dtype after normalize:", rep["timestamp"].dtype)

    t1 = perf_counter()

    rep = rep.copy()

    # Scale energy by number of turbines to represent park-level production.
    n_turb = max(1, int(req.n_turbines))
    rep["energy_kwh"] = rep["energy_kwh"].astype(float) * n_turb

    # Aggregate to month-start totals and align to the profit horizon index.
    monthly_kwh = (
        rep.assign(month=lambda df: df["timestamp"].dt.to_period("M").dt.start_time)
        .groupby("month")["energy_kwh"]
        .sum()
        .sort_index()
        .astype(float)
    )
    monthly_mwh = (monthly_kwh / 1000.0).reindex(target_index)

    # Diagnostic: identify any missing months relative to the target horizon.
    missing = target_index.difference(monthly_mwh.dropna().index)
    print("[FORECAST] target months:", len(target_index))
    print("[FORECAST] produced months:", monthly_mwh.notna().sum())
    print("[FORECAST] first/last rep ts:", rep["timestamp"].min(), rep["timestamp"].max())
    print("[FORECAST] missing months (first 10):", list(missing[:10]))

    # Fill gaps (if any) to ensure full 20-year profit model coverage.
    if monthly_mwh.isna().any():
        monthly_mwh = monthly_mwh.ffill()
        if monthly_mwh.isna().any():
            raise ValueError("Forecast did not produce enough months to cover the profit horizon.")

    t2 = perf_counter()
    print(
        f"[TIMING] forecast hist+mc={t1-t0:.3f}s | align={t2-t1:.3f}s | "
        f"fast={forecast_fast} | era5_frac={era5_frac} | n_sims={n_sims}"
    )

    # Return series explicitly indexed by the target horizon for downstream code stability.
    mwh_series = pd.Series(monthly_mwh.values, index=target_index)
    return mwh_series, rep


@app.post("/api/calc")
def calc(req: CalcRequest):
    """
    Main calculation endpoint.

    Flow:
      1) Build a monthly MWh series:
         - forecast mode when latitude/longitude are provided
         - constant fallback otherwise
      2) Run financial model (power_to_profit)
      3) Build summary payload with quantiles, tables, and optional report email
    """
    try:
        # Normalize COD to month-start for consistent indexing.
        cod = pd.Timestamp(req.cod_date)
        cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

        # Forecast path is only available when coordinates are provided.
        use_forecast = (req.latitude is not None) and (req.longitude is not None)

        rep_df = None
        if use_forecast:
            mwh_series, rep_df = build_mwh_series_from_forecast(req)
            production_mode = "forecast"
        else:
            # Constant-production fallback for demo/testing without coordinates.
            dates = pd.date_range(cod, periods=int(FORECAST_MONTHS), freq="MS")
            mwh_series = pd.Series(
                float(DEFAULT_MWH_PER_TURBINE_PER_MONTH) * float(req.n_turbines),
                index=dates
            )
            production_mode = "constant"

        # Serialize monthly MWh series for JSON responses (month_start as date string).
        mwh_monthly_used = [
            {"month_start": str(idx.date()), "mwh": float(val)}
            for idx, val in mwh_series.items()
        ]

        # Derive annual totals for convenience in the report/UI.
        mwh_yearly_used = (
            mwh_series.to_frame("mwh")
            .assign(calendar_year=lambda df: df.index.year)
            .groupby("calendar_year", as_index=False)["mwh"]
            .sum()
            .to_dict(orient="records")
        )

        # Revenue model configuration passed through to the profit model.
        revenue_kwargs = {
            "tso_id": int(req.tso_id),
            "eeg_on": 1 if req.eeg_on else 0,
            "cod_date": req.cod_date,
        }

        # Skip heavy monthly revenue outputs when fast_mode is enabled.
        include_monthly = not bool(req.fast_mode)

        # Run profit Monte Carlo / financial model.
        result = power_to_profit(
            mwh_monthly_20y=mwh_series,
            turbine_type_id=req.turbine_type_id,
            n_turbines=req.n_turbines,
            hub_height_m=req.hub_height_m,
            equity_eur=float(req.equity_eur),
            revenue_kwargs=revenue_kwargs,
            include_monthly_revenue=include_monthly,
        )

        # Extract standard outputs. DataFrames may be empty depending on model settings.
        final_summary_df = result.get("final_summary_df", pd.DataFrame())
        stats_df = result.get("stats_df", pd.DataFrame())

        # Pull metric quantiles for headline KPIs.
        npv_summary = pick_summary_row(final_summary_df, "NPV (€)")
        irr_summary = pick_summary_row(final_summary_df, "IRR")

        # Breakdown summaries from the model (may vary by model version).
        capex = result.get("capex_summary", {}) or {}
        fin = result.get("finance_summary", {}) or {}
        opex = result.get("opex_summary", {}) or {}

        # yearly_df_by_sim contains per-simulation annual results; used to compute quantiles by year.
        yearly = result["yearly_df_by_sim"].copy()
        yearly["project_year"] = yearly["project_year"].astype(int)

        # Deterministic per-year columns taken from the first simulation row per year.
        det_cols = yearly.groupby("project_year", as_index=False).agg(
            year_start=("year_start", "first"),
            opex=("annual_opex_eur", "first"),
            debt_service=("debt_service_eur", "first"),
        )

        # Revenue quantiles per year (P10/P50/P90).
        rev_q = (
            yearly.groupby("project_year")["revenue_eur"]
            .quantile([0.10, 0.50, 0.90])
            .unstack()
            .rename(columns={0.10: "revenue_p10", 0.50: "revenue_p50", 0.90: "revenue_p90"})
            .reset_index()
        )

        # Profit quantiles per year (after OPEX and debt service).
        profit_q = (
            yearly.groupby("project_year")["profit_after_opex_and_debt_eur"]
            .quantile([0.10, 0.50, 0.90])
            .unstack()
            .rename(columns={0.10: "profit_p10", 0.50: "profit_p50", 0.90: "profit_p90"})
            .reset_index()
        )

        # Build yearly table used by charts and the PDF.
        table = (
            det_cols
            .merge(rev_q, on="project_year", how="left")
            .merge(profit_q, on="project_year", how="left")
            .rename(columns={"project_year": "year"})
        )

        # Normalize date formatting to ISO string for JSON safety.
        if "year_start" in table.columns:
            table["year_start"] = pd.to_datetime(table["year_start"]).dt.strftime("%Y-%m-%d")

        # Replace inf and NaN to keep JSON strict.
        table = table.replace([np.inf, -np.inf], np.nan).where(pd.notnull(table), None)

        # Assemble response payload. This payload also feeds the PDF report builder.
        payload = {
            "ok": True,
            "env": APP_ENV,

            "latitude": req.latitude,
            "longitude": req.longitude,
            "email": req.email,
            "fast_mode": req.fast_mode,
            "production_mode": production_mode,
            "mwh_monthly_used": mwh_monthly_used,
            "mwh_yearly_used": mwh_yearly_used,

            "npv_eur": npv_summary,
            "irr": irr_summary,

            "eeg_strike_eur_per_mwh": result.get("eeg_strike_eur_per_mwh"),
            "eeg_uplift_total_eur_mean": result.get("eeg_uplift_total_eur_mean"),
            "eeg_uplift_total_eur_p50": result.get("eeg_uplift_total_eur_p50"),

            "discount_rate_used": float(result.get("discount_rate_used")),
            "wacc": float(fin.get("wacc")) if fin.get("wacc") is not None else None,
            "n_sims": int(result.get("n_sims", 0)),
            "terminal_value_eur": float(result.get("terminal_value_eur")),

            "capex_total_eur": float(capex.get("total_capex_eur")) if capex.get("total_capex_eur") is not None else None,
            "park_mw": float(capex.get("park_mw")) if capex.get("park_mw") is not None else None,

            "equity_eur": float(fin.get("equity_eur")) if fin.get("equity_eur") is not None else float(req.equity_eur),
            "debt_eur": float(fin.get("debt_eur")) if fin.get("debt_eur") is not None else None,
            "equity_share": float(fin.get("equity_share_derived")) if fin.get("equity_share_derived") is not None else None,

            "opex_total_eur": float(opex.get("total_opex_eur")) if opex.get("total_opex_eur") is not None else None,

            "final_summary_table": to_jsonable(final_summary_df),
            "stats_table": to_jsonable(stats_df),
            "yearly_table": table.to_dict(orient="records"),

            "turbine_model": f"Type {req.turbine_type_id}",

            # equity_share appears twice; keep as-is for backward compatibility with any consumers.
            "equity_share": float(fin.get("equity_share_derived")) if fin.get("equity_share_derived") is not None else None,

            "eeg_on": bool(req.eeg_on),

            "eeg_uplift_total_eur": result.get("eeg_uplift_total_eur_mean"),
        }

        # Include representative forecast path for debugging/visualization when forecast mode is used.
        if rep_df is not None:
            rep_out = rep_df.copy()
            rep_out["timestamp"] = pd.to_datetime(rep_out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            payload["forecast_rep_path"] = rep_out.to_dict(orient="records")

        # Include monthly simulation data only when requested (non-fast mode).
        if include_monthly and result.get("monthly_revenue_df_by_sim") is not None:
            payload["monthly_revenue_by_sim"] = to_jsonable(result["monthly_revenue_df_by_sim"])

        # Final cleanup and strict JSON validation.
        payload = scrub_nonfinite(payload)
        json.dumps(payload, allow_nan=False)

        # Email sending is gated by presence of email and non-fast mode (PDF generation is heavier).
        if req.email and (not req.fast_mode):
            pdf_bytes = build_report_pdf_bytes(
                project_meta={
                    "cod_date": req.cod_date,
                    "n_turbines": req.n_turbines,
                    "hub_height_m": req.hub_height_m,
                    "turbine_type_id": req.turbine_type_id,
                    "fast_mode": req.fast_mode,
                    "lat": req.latitude,
                    "lon": req.longitude,
                    "rated_power_mw": payload.get("park_mw"),
                    "start_year": pd.Timestamp(req.cod_date).year,
                    "end_year": pd.Timestamp(req.cod_date).year + 20,
                    "report_date": date.today().isoformat(),
                },
                payload=payload,
            )

            send_calc_email(
                to_email=req.email,
                project_meta={
                    "cod_date": req.cod_date,
                    "n_turbines": req.n_turbines,
                    "hub_height_m": req.hub_height_m,
                    "turbine_type_id": req.turbine_type_id,
                    "fast_mode": req.fast_mode,
                },
                result_meta={
                    "n_sims": payload.get("n_sims"),
                    "discount_rate_used": payload.get("discount_rate_used"),
                },
                pdf_bytes=pdf_bytes,
                pdf_filename="wind_park_report.pdf",
            )

        return payload

    except Exception:
        # Surface full traceback for easier debugging during integration.
        # HTTP 400 indicates invalid inputs or model execution issues.
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=tb)


# ============================================================
# Local runner
# ============================================================
if __name__ == "__main__":
    # Uvicorn is used for local execution. In production, a process manager may launch the app.
    import uvicorn

    host = "127.0.0.1" if IS_LOCAL else SERVER_HOST
    port = LOCAL_PORT if IS_LOCAL else SERVER_PORT

    # reload=IS_LOCAL enables code reload on file changes for development only.
    uvicorn.run("main:app", host=host, port=port, reload=IS_LOCAL)
