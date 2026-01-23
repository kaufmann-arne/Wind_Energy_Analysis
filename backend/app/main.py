from __future__ import annotations
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from datetime import date, datetime, timezone
import os
import json
import math
import sys
import hashlib
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from time import perf_counter
import traceback


import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# PATH SETUP
# project/backend/app/main.py -> project/backend
# ============================================================
BACKEND_ROOT = Path(__file__).resolve().parents[1]   # .../project/backend

# Allow importing backend/model (profit model code)
MODEL_DIR = BACKEND_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

# ML subproject root:
# backend/wind-power-climate-ml/
ML_ROOT = BACKEND_ROOT / "wind-power-climate-ml"
ML_SRC = ML_ROOT / "src"
sys.path.insert(0, str(ML_SRC))  # enables: from ml.production_forecast import ...


# ============================================================
# Imports that depend on PATH SETUP
# ============================================================
from profit_mc import power_to_profit, FORECAST_MONTHS  # noqa: E402

from ml.production_forecast import (  # noqa: E402
    build_hist_monthly_energy_lookup,
    forecast_monthly_representative_path_from_hist_energy,
)


# ============================================================
# ENV SWITCH: local vs server
# ============================================================
APP_ENV = os.getenv("APP_ENV", "server").lower()
IS_LOCAL = APP_ENV in ("local", "dev", "development")

LOCAL_PORT = int(os.getenv("LOCAL_PORT", "5137"))
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

app = FastAPI()


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/")
def root():
    return {
        "ok": True,
        "env": APP_ENV,
        "message": "Backend running. Go to /docs for API docs.",
    }


# ============================================================
# CORS
# ============================================================
if IS_LOCAL:
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5137",
        "http://127.0.0.1:5137",
    ]
else:
    allow_origins = ["*"]  # tighten later if needed

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
DEFAULT_ERA5_DIR = Path(
    os.getenv("ERA5_DIR", str(ML_ROOT / "data" / "raw" / "era5"))
)
DEFAULT_TURBINE_CURVES = Path(
    os.getenv("TURBINE_CURVES_JSON", str(ML_ROOT / "data" / "turbine_power_curves.json"))
)
DEFAULT_MODEL_DIR = Path(
    os.getenv("FORECAST_MODEL_DIR", str(ML_ROOT / "LGBM_Model" / "model_artifacts"))
)

DEFAULT_MWH_PER_TURBINE_PER_MONTH = 1200.0


# Optional sanity checks (recommended during development)
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
    model_config = ConfigDict(extra="ignore")

    # Location + meta
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    email: Optional[str] = None

    # UI settings
    n_turbines: int = Field(default=5, ge=1)
    hub_height_m: int = Field(default=160, ge=1)

    # Profit model turbine selection (numeric)
    turbine_type_id: int = Field(default=1, ge=0, le=2)

    # Forecast speed:
    # True => fast forecast; False => slow forecast
    fast_mode: Optional[bool] = None

    equity_eur: int = Field(default=1_000_000, ge=0)

    # Revenue settings
    tso_id: int = Field(default=1, ge=0, le=3)
    eeg_on: bool = True
    cod_date: str  # "YYYY-MM-DD"

# ============================================================
# Email helper
# ============================================================
def _parse_iso_dt(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _val(d: dict, key: str, placeholder: str) -> Any:
    v = d.get(key, None)
    return placeholder if v is None or v == "" else v


def _fmt_num(x: Any, placeholder: str) -> str:
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
    try:
        return date.today().isoformat()
    except Exception:
        return "[REPORT_DATE]"


def _short_label(s: str, max_len: int = 14) -> str:
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
    Input monthly:
      [{"timestamp":"...Z","energy_kwh":...}, ...]

    Returns:
      monthly_mwh: [{"month_start":"YYYY-MM-01","mwh":...}, ...]
      annual_mwh:  [{"year":YYYY,"mwh":...}, ...]
    """
    if not energy_forecast:
        return [], []

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

    monthly_mwh: List[Dict[str, Any]] = []
    for dt, mwh in rows:
        month_start = f"{dt.year:04d}-{dt.month:02d}-01"
        monthly_mwh.append({"month_start": month_start, "mwh": mwh})

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
    AEP: average annual MWh. Prefer complete years; else scale partial years to annual equivalent.
    CF: AEP / (rated_power_mw * 8760) * 100
    """
    if not monthly_mwh or not rated_power_mw:
        return None, None

    by_year_sum: Dict[int, float] = {}
    by_year_months: Dict[int, set] = {}

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

    full_years = [y for y in by_year_months if len(by_year_months[y]) == 12]
    if full_years:
        aep = sum(by_year_sum[y] for y in full_years) / float(len(full_years))
    else:
        scaled = []
        for y in sorted(by_year_sum.keys()):
            m_count = max(1, len(by_year_months.get(y, set())))
            scaled.append(by_year_sum[y] * (12.0 / m_count))
        aep = sum(scaled) / float(len(scaled)) if scaled else None

    cf = (aep / (rated_power_mw * 8760.0)) * 100.0 if (aep is not None and rated_power_mw) else None
    return aep, cf


# ============================
# PDF builder (4 pages)
# ============================

intro_text = (
    "This report provides an automated, site-specific assessment of expected wind energy production "
    "and project economics based on the inputs selected on the website. It combines a long-term "
    "energy forecast at monthly resolution with a financial model that translates production into "
    "revenues, costs, and value metrics such as NPV and IRR. All figures and tables are indicative and intended for "
    "early-stage screening; final investment decisions require detailed engineering, permitting, "
    "and commercial due diligence."
)


def build_report_pdf_bytes(*, project_meta: dict, payload: dict,  logo_path: str = str(DEFAULT_LOGO_PATH)) -> bytes:
    """
    Layout fixes:
      - Smaller charts (page 2/3/4) and less "after chart" whitespace
      - OPEX table no longer squeezed into footer
      - Decision summary no longer collides with disclaimer/footer
      - Annual energy year axis uses integer ticks
      - CAPEX chart uses labels (not "category index")
    """

    # ---------- chart sizing knobs ----------
    H_ENERGY = 300   # page 2 chart height (was 380+)
    H_PRICE  = 200   # page 3
    H_CAPEX  = 190   # page 3
    H_CUMCF  = 190   # page 4
    H_TORN   = 190   # page 4
    CHART_GAP_AFTER = 15  # reduce from ~18 (less bottom waste)

    # ---------- helper: chart -> PNG bytes ----------
    def fig_to_png_bytes(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ---------- parse rated power ----------
    rated_power_mw = project_meta.get("rated_power_mw", None)
    try:
        rated_power_mw = float(rated_power_mw) if rated_power_mw is not None else None
    except Exception:
        rated_power_mw = None

    # ---------- normalize energy inputs ----------
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
    

    print("[PDF] annual_mwh_series len:", len(annual_mwh_series))
    print("[PDF] annual_mwh_series first rows:", annual_mwh_series[:3])



    if monthly_mwh and rated_power_mw:
        aep, cf = derive_aep_and_capacity_factor(monthly_mwh, rated_power_mw)
        payload.setdefault("metrics", {})
        payload["metrics"].setdefault("aep_mwh_per_year", aep)
        payload["metrics"].setdefault("capacity_factor_pct", cf)
    def _derive_seasonality_and_variability(monthly_mwh: List[Dict[str, Any]]):
        # month_start: "YYYY-MM-01"
        by_month = {m: [] for m in range(1, 13)}
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
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            ax.tick_params(axis="both", labelsize=8)
            chart1_png = fig_to_png_bytes(fig1)

    seasonality, min_a, mean_a, max_a, cv_pct = _derive_seasonality_and_variability(monthly_mwh)

    chart_season_png = None
    if seasonality and any(v is not None for v in seasonality):
        # replace Nones with 0 for plotting
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
    yearly_tbl = payload.get("yearly_table") or []
    rev_x, rev_p10, rev_p50, rev_p90 = [], [], [], []

    for r in yearly_tbl:
        try:
            y = int(r.get("year"))
        except Exception:
            continue
        # allow None safely
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
    capex_total = payload.get("capex_total_eur", None)
    opex_total = payload.get("opex_total_eur", None)

    def _to_float(x):
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
    styles = getSampleStyleSheet()
    style_body = styles["BodyText"]
    style_body.fontName = "Helvetica"
    style_body.fontSize = 10
    style_body.leading = 13

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    MARGIN_L = 40
    MARGIN_R = 40
    TOP = height - 40
    FOOT = 35

    def draw_header(page_no: int):
        c.setFont("Helvetica", 10)
        c.drawString(MARGIN_L, TOP, "Wind Site Assessment - Summary Report")
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(1)
        c.line(MARGIN_L, TOP - 12, width - MARGIN_R, TOP - 12)

    def draw_footer(page_no: int):
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(1)
        c.line(MARGIN_L, FOOT + 18, width - MARGIN_R, FOOT + 18)
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(MARGIN_L, FOOT, "All bracketed values [LIKE_THIS] are populated by the application.")
        c.drawRightString(width - MARGIN_R, FOOT, f"Page {page_no}")
        c.setFillColor(colors.black)

    def draw_paragraph(text: str, x: float, y: float, w: float) -> float:
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
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, title)
        y -= 12

        c.setStrokeColor(colors.HexColor("#999999"))
        c.rect(x, y - h, w, h, stroke=1, fill=0)

        if png_bytes:
            img = ImageReader(io.BytesIO(png_bytes))
            c.drawImage(img, x + 6, y - h + 6, width=w - 12, height=h - 12,
                        preserveAspectRatio=True, mask="auto")
        else:
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.grey)
            c.drawString(x + 12, y - 22, "CHART PLACEHOLDER - Data not available")
            c.setFillColor(colors.black)

        return y - h - CHART_GAP_AFTER

    # ============================
    # Page 1: De-cluttered summary (unchanged core, already compact)
    # ============================
    draw_header(1)

    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN_L, height - 130, "Wind Site Assessment & Financial Summary")
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN_L, height - 155, "Automated report generated from website outputs (4 pages).")

    # NEW: intro paragraph below title
    intro_y = height - 175
    intro_y = draw_paragraph(intro_text, MARGIN_L, intro_y, width - MARGIN_L - MARGIN_R)

    # Start the two-column tables below the intro with some spacing
    y_top = intro_y - 18


    gutter = 18
    col_w = (width - MARGIN_L - MARGIN_R - gutter) / 2
    x_left = MARGIN_L
    x_right = MARGIN_L + col_w + gutter

    equity_share = payload.get("equity_share", None)
    eeg_support = payload.get("eeg_on", None)
    eeg_uplift_total = payload.get("eeg_uplift_total_eur", None)
    lat = _val(project_meta, "lat", _val(payload, "latitude", "[LAT]"))
    lon = _val(project_meta, "lon", _val(payload, "longitude", "[LON]"))

    turbine_model = payload.get("turbine_model") or _val(project_meta, "turbine_model", "[TURBINE_MODEL]")
    rated_power_txt = _val(project_meta, "rated_power_mw", "[RATED_POWER_MW]")
    hub_height = _val(project_meta, "hub_height_m", "[HUB_HEIGHT_M]")
    # --- Horizon: derive from COD if start/end not provided ---
    cod = project_meta.get("cod_date") or payload.get("cod_date")
    # cod may be "YYYY-MM-DD" or Timestamp-like; normalize to string
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
    npv_q = payload.get("npv_eur", {}) or {}
    irr_q = payload.get("irr", {}) or {}

    def _irr_to_pct(x):
        try:
            return float(x) * 100.0 if x is not None else None
        except Exception:
            return None

    best_npv = npv_q.get("p90", None)
    base_npv = npv_q.get("mean", None)   # you requested MEAN for base
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

    y = min(y_left_end, y_right_end) - 16

    metrics = payload.get("metrics", {}) or {}

    aep = _val(metrics, "aep_mwh_per_year", "[AEP_MWH_PER_YEAR]")
    cf = _val(metrics, "capacity_factor_pct", "[CAPACITY_FACTOR_PCT]")
    payback = int(end_year) - int(start_year) if str(start_year).isdigit() and str(end_year).isdigit() else 20

    # --- NPV/IRR: use MEAN from payload summaries (not p50) ---
    npv_mean = payload.get("npv_eur", {}).get("mean", None)

    # irr in your payload is typically a fraction (e.g., 0.08 = 8%)
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

    y = draw_chart("Figure 2 - Annual Revenue (P10/P50/P90)",
               chart2_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_PRICE)


    # KPI strip under chart (IMPORTANT: update y)
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

    # ---- Seasonality + variability block ----
    y -= 14
    gutter = 18
    col_w = (width - MARGIN_L - MARGIN_R - gutter) / 2
    x_left = MARGIN_L
    x_right = MARGIN_L + col_w + gutter

    y_left = draw_chart(
        "Seasonality - Average Monthly Energy",
        chart_season_png,
        x_left,
        y,
        col_w,
        140
    )

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

    y = draw_chart("Figure 2 - Power Price Path (Best/Base/Worst)",
                   chart2_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_PRICE)

    y = draw_chart("Figure 3 - Total Costs (CAPEX vs OPEX)",
               chart3_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_CAPEX)


    # OPEX (give it room and slightly smaller font)
    y -= 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN_L, y, "OPEX Summary (Base Case)")
    y -= 10

    # --- OPEX Summary from payload (since assumptions.opex is not returned) ---
    yearly_tbl = payload.get("yearly_table") or []

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

    y = draw_chart("Figure 4 - Cumulative Profit (P50, not discounted)",
               chart4_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_CUMCF)

    y = draw_chart("Figure 5 - NPV Uncertainty (P10/Mean/P90)",
               chart5_png, MARGIN_L, y, width - MARGIN_L - MARGIN_R, H_TORN)


    y -= 6
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
    Sends a notification email when a detailed (non-fast) calculation was completed."""


    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "")

    if not smtp_host or not smtp_from:
        print("[email] SMTP not configured. Skipping email.")
        return

    use_tls = os.getenv("SMTP_TLS", "true").strip().lower() in ("1", "true", "yes", "on")

    msg = EmailMessage()
    msg["Subject"] = "Your wind park analysis is ready"
    msg["From"] = smtp_from
    msg["To"] = to_email

    body = (
        "Dear Sir or Madam,\n\n"
        "Thank you for using our service.\n\n"
        "We are pleased to inform you that the analysis of your wind park project "
        "has been completed successfully.\n\n"
        "Please find attached a detailed PDF report containing all relevant metrics "
        "and key data related to your project.\n\n"
        "You may now return to the application to review the results in more detail.\n\n"
        "Should you have any questions or require further assistance, please do not "
        "hesitate to contact us.\n\n"
        "Kind regards,\n"
        "Wind Model Backend Team\n"
    )
    msg.set_content(body)

    # Attach PDF (optional)
    if pdf_bytes:
        msg.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename=pdf_filename,
        )


    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            if use_tls:
                server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            print(f"[email] Sent calculation email to {to_email}")
    except Exception as e:
        print(f"[email] Failed to send email to {to_email}: {e}")


# ============================================================
# JSON helpers (make pandas/numpy safe for JSON)
# ============================================================
def to_jsonable(obj: Any):
    """Convert pandas/numpy objects into JSON-serializable types + replace NaN/Inf with None."""
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
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
    """Recursively replace NaN/Inf/-Inf with None in already JSON-like objects."""
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
    final_summary_df format:
      metric | mean | p10 | p50 | p90
    Returns a dict {mean,p10,p50,p90} for the requested metric row.
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
    """Deterministic seed per (location, turbine)."""
    s = f"{lat:.6f}|{lon:.6f}|{hub_height_m:.3f}|{turbine_type}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)



def build_mwh_series_from_forecast(req: CalcRequest) -> tuple[pd.Series, pd.DataFrame]:
    """
    Runs the forecast and returns:
      - monthly MWh pd.Series indexed by month-start (MS) covering FORECAST_MONTHS from COD
      - rep dataframe (useful if you want to return it)
    Key change vs your old forecast API:
      - We align forecast to COD YEAR+MONTH (not just year).
      - Under the hood the forecast function starts at start_year=COD.year,
        then we SHIFT timestamps by (COD.month-1) months.
    """
    if req.latitude is None or req.longitude is None:
        raise ValueError("latitude/longitude required for forecast mode")

    # Decide forecast speed based on fast_mode 
    forecast_fast = bool(req.fast_mode)

    cod = pd.Timestamp(req.cod_date)
    cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)
    horizon_months = int(FORECAST_MONTHS)
    target_index = pd.date_range(cod, periods=horizon_months, freq="MS")


    # Forecast params
    seed = stable_seed(
        lat=float(req.latitude),
        lon=float(req.longitude),
        hub_height_m=float(req.hub_height_m),
        turbine_type=req.turbine_type_id,
    )
    era5_frac = 0.10 if forecast_fast else 1.0
    n_sims = 150 if forecast_fast else 500
    era5_subsample_seed = 42
    mc_seed = seed



    # Run forecast
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
        start_date=req.cod_date,   # function starts at year boundary
        years=int(math.ceil(horizon_months / 12)),  # ensure we cover the horizon
        n_sims=n_sims,
        random_seed=mc_seed,
    )
    # Normalize timestamps: make them UTC then tz-naive so they match target_index
    rep["timestamp"] = pd.to_datetime(rep["timestamp"], utc=True).dt.tz_convert(None)
    print("[FORECAST] ts dtype after normalize:", rep["timestamp"].dtype)


    t1 = perf_counter()

    rep = rep.copy()

    # Scale to park size (if forecast is per-turbine)
    n_turb = max(1, int(req.n_turbines))
    rep["energy_kwh"] = rep["energy_kwh"].astype(float) * n_turb


    # Make monthly series (sum per month) and align to target horizon
    monthly_kwh = (
        rep.assign(month=lambda df: df["timestamp"].dt.to_period("M").dt.start_time)
        .groupby("month")["energy_kwh"]
        .sum()
        .sort_index()
        .astype(float)
    )
    monthly_mwh = (monthly_kwh / 1000.0).reindex(target_index)

    missing = target_index.difference(monthly_mwh.dropna().index)
    print("[FORECAST] target months:", len(target_index))
    print("[FORECAST] produced months:", monthly_mwh.notna().sum())
    print("[FORECAST] first/last rep ts:", rep["timestamp"].min(), rep["timestamp"].max())
    print("[FORECAST] missing months (first 10):", list(missing[:10]))


    # Fill if needed (policy choice)
    if monthly_mwh.isna().any():
        monthly_mwh = monthly_mwh.ffill()
        if monthly_mwh.isna().any():
            raise ValueError("Forecast did not produce enough months to cover the profit horizon.")

    t2 = perf_counter()
    print(
        f"[TIMING] forecast hist+mc={t1-t0:.3f}s | align={t2-t1:.3f}s | "
        f"fast={forecast_fast} | era5_frac={era5_frac} | n_sims={n_sims}"
    )

    mwh_series = pd.Series(monthly_mwh.values, index=target_index)
    return mwh_series, rep


@app.post("/api/calc")
def calc(req: CalcRequest):
    try:
        # ---------------------------------------------------------
        # 1) Build monthly MWh series for the model horizon (20y)
        #    - Use forecast if lat/lon provided
        #    - Else fallback to constant mode
        # ---------------------------------------------------------
        cod = pd.Timestamp(req.cod_date)
        cod = pd.Timestamp(year=cod.year, month=cod.month, day=1)

        use_forecast = (req.latitude is not None) and (req.longitude is not None)

        rep_df = None
        if use_forecast:
            mwh_series, rep_df = build_mwh_series_from_forecast(req)
            production_mode = "forecast"
        else:
            dates = pd.date_range(cod, periods=int(FORECAST_MONTHS), freq="MS")
            mwh_series = pd.Series(
                float(DEFAULT_MWH_PER_TURBINE_PER_MONTH) * float(req.n_turbines),
                index=dates
            )
            production_mode = "constant"

        # ---------------------------------------------------------
        # 1b) PRODUCTION DEBUG OUTPUTS (what we actually used)
        # ---------------------------------------------------------
        mwh_monthly_used = [
            {"month_start": str(idx.date()), "mwh": float(val)}
            for idx, val in mwh_series.items()
        ]

        mwh_yearly_used = (
            mwh_series.to_frame("mwh")
            .assign(calendar_year=lambda df: df.index.year)
            .groupby("calendar_year", as_index=False)["mwh"]
            .sum()
            .to_dict(orient="records")
        )


        # ---------------------------------------------------------
        # 2) Revenue kwargs (paths are hardcoded inside revenue.py)
        # ---------------------------------------------------------
        revenue_kwargs = {
            "tso_id": int(req.tso_id),
            "eeg_on": 1 if req.eeg_on else 0,  # profit_mc expects int 0/1
            "cod_date": req.cod_date,
        }

        # "precise" => include monthly revenue output (bigger output)
        include_monthly = not bool(req.fast_mode)

        result = power_to_profit(
            mwh_monthly_20y=mwh_series,
            turbine_type_id=req.turbine_type_id,
            n_turbines=req.n_turbines,
            hub_height_m=req.hub_height_m,
            equity_eur=float(req.equity_eur),
            revenue_kwargs=revenue_kwargs,
            include_monthly_revenue=include_monthly,
        )

        # ---------------------------------------------------------
        # 3) Build response payload
        # ---------------------------------------------------------
        final_summary_df = result.get("final_summary_df", pd.DataFrame())
        stats_df = result.get("stats_df", pd.DataFrame())

        npv_summary = pick_summary_row(final_summary_df, "NPV (€)")
        irr_summary = pick_summary_row(final_summary_df, "IRR")

        capex = result.get("capex_summary", {}) or {}
        fin = result.get("finance_summary", {}) or {}
        opex = result.get("opex_summary", {}) or {}

        # ---------------------------------------------------------
        # 4) Yearly P10/P50/P90 table (across sims) ✅ FIXED (wide quantiles)
        # ---------------------------------------------------------
        yearly = result["yearly_df_by_sim"].copy()
        yearly["project_year"] = yearly["project_year"].astype(int)

        # deterministic per-year cols (same across sims, so take first)
        det_cols = yearly.groupby("project_year", as_index=False).agg(
            year_start=("year_start", "first"),
            opex=("annual_opex_eur", "first"),
            debt_service=("debt_service_eur", "first"),
        )

        # quantiles in WIDE format (no "level_1" column)
        rev_q = (
            yearly.groupby("project_year")["revenue_eur"]
            .quantile([0.10, 0.50, 0.90])
            .unstack()
            .rename(columns={0.10: "revenue_p10", 0.50: "revenue_p50", 0.90: "revenue_p90"})
            .reset_index()
        )

        profit_q = (
            yearly.groupby("project_year")["profit_after_opex_and_debt_eur"]
            .quantile([0.10, 0.50, 0.90])
            .unstack()
            .rename(columns={0.10: "profit_p10", 0.50: "profit_p50", 0.90: "profit_p90"})
            .reset_index()
        )

        table = (
            det_cols
            .merge(rev_q, on="project_year", how="left")
            .merge(profit_q, on="project_year", how="left")
            .rename(columns={"project_year": "year"})
        )

        # ensure year_start is string and stable
        if "year_start" in table.columns:
            table["year_start"] = pd.to_datetime(table["year_start"]).dt.strftime("%Y-%m-%d")

        table = table.replace([np.inf, -np.inf], np.nan).where(pd.notnull(table), None)


        payload = {
            "ok": True,
            "env": APP_ENV,

            # echo request meta
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

            # turbine model (string)
            "turbine_model": f"Type {req.turbine_type_id}",  # replace with your real mapping if you have one

            # equity share (already there, keep it and make sure it is in payload)
            "equity_share": float(fin.get("equity_share_derived")) if fin.get("equity_share_derived") is not None else None,

            # EEG applied flag
            "eeg_on": bool(req.eeg_on),

            # EEG uplift total (mean)
            "eeg_uplift_total_eur": result.get("eeg_uplift_total_eur_mean"),

        }

        # Optional: return forecast rep path too (can be useful; not huge)
        if rep_df is not None:
            rep_out = rep_df.copy()
            rep_out["timestamp"] = pd.to_datetime(rep_out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            payload["forecast_rep_path"] = rep_out.to_dict(orient="records")

        # Optional: if you enabled include_monthly_revenue, you might want to return it.
        # WARNING: can be huge.
        if include_monthly and result.get("monthly_revenue_df_by_sim") is not None:
            payload["monthly_revenue_by_sim"] = to_jsonable(result["monthly_revenue_df_by_sim"])

        payload = scrub_nonfinite(payload)
        json.dumps(payload, allow_nan=False)  # strict JSON

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
                    # optional but recommended:
                    "rated_power_mw": payload.get("park_mw"),   # or compute if per turbine
                    "start_year": pd.Timestamp(req.cod_date).year,
                    "end_year": pd.Timestamp(req.cod_date).year + 20,  # horizon assumption
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

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=tb)


# ============================================================
# Local runner
# ============================================================
if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1" if IS_LOCAL else SERVER_HOST
    port = LOCAL_PORT if IS_LOCAL else SERVER_PORT

    uvicorn.run("main:app", host=host, port=port, reload=IS_LOCAL)
