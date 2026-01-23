# Wind Park Profitability and Risk Assessment Web Application

## Academic Project README

---

## 1. Introduction

This repository contains a full-stack web application developed for the **academic analysis of onshore wind park feasibility**. The system integrates geospatial user input, climate-based wind power forecasting, and a stochastic financial model to provide **quantitative estimates of energy production, revenues, profitability, and financial risk** over a long-term project horizon.

The application is intended as a **screening and exploratory decision-support tool**. It supports comparative analysis across locations and assumptions, and emphasizes **uncertainty-aware outputs** (e.g., P10 / P50 / P90 quantiles) rather than deterministic forecasts.

This project is suitable for use in:
- Energy systems research
- Renewable energy economics
- Infrastructure finance and policy analysis
- Teaching and demonstration of integrated techno-economic models

It is **not** intended to replace site-specific measurements, engineering studies, or professional investment due diligence.

---

## 2. System Overview

### 2.1 Architectural Concept

The system follows a **decoupled client–server architecture**, where the frontend and backend are developed, deployed, and executed independently.

```
+------------------+        HTTP (JSON)        +--------------------------+
|   Frontend UI   |  ----------------------> |   Backend API            |
|  React + TS     |                           |  FastAPI (Python)        |
+------------------+                           +--------------------------+
        |                                                     |
        | User input, maps                                   | Wind forecast
        | Result visualization                               | Financial model
        |                                                     | PDF generation
```

### 2.2 Design Rationale

- **Loose coupling** allows independent scaling and deployment
- **Stateless API** simplifies reproducibility and testing
- **Client-side geospatial logic** reduces backend complexity
- **Monte-Carlo simulation** enables explicit modeling of uncertainty

---

## 3. Frontend Component

### 3.1 Technology Stack

- React (functional components)
- TypeScript
- Leaflet / React-Leaflet (mapping)
- Fetch API for backend communication

### 3.2 Responsibilities

The frontend is responsible for:
- Collecting user-defined project assumptions
- Providing map-based spatial interaction
- Automatically detecting the applicable TSO zone
- Submitting calculation requests to the backend
- Rendering tabular, graphical, and KPI-based results

### 3.3 Geospatial Processing

The application performs **client-side point-in-polygon tests** using GeoJSON data representing German TSO zones. The implementation:
- Supports `Polygon` and `MultiPolygon` geometries
- Handles overlapping zones by selecting the smallest enclosing polygon
- Avoids external GIS dependencies

This ensures deterministic mapping from geographic coordinates to `tso_id` values expected by the backend.

---

## 4. Backend Component

### 4.1 Technology Stack

- Python 3
- FastAPI
- NumPy, Pandas
- Matplotlib (headless)
- ReportLab (PDF generation)
- SMTP (optional email delivery)

### 4.2 Core Responsibilities

The backend performs:
- Wind energy production forecasting
- Conversion of energy output into revenues
- Monte-Carlo–based financial modeling
- Aggregation of uncertainty metrics
- Generation of structured JSON responses
- Optional creation and emailing of a multi-page PDF report

---

## 5. Wind Energy Production Forecast

### 5.1 Data Sources

- ERA5 reanalysis wind data
- Turbine-specific power curves
- Pre-trained machine learning model artifacts

### 5.2 Forecast Methodology

For a given location, hub height, and turbine type:
1. Historical wind conditions are sampled
2. A representative monthly production path is generated
3. Production is aligned to the **Commissioning Date (COD)**
4. Monthly energy is scaled by the number of turbines

Two modes are supported:

| Mode | Description |
|----|------------|
| Fast | Reduced sampling, faster response |
| Precise | Higher resolution, more Monte-Carlo simulations |

---

## 6. Financial Model

### 6.1 Model Structure

The financial model converts energy production into cash flows using:
- Market electricity prices (by TSO)
- EEG support mechanisms (optional)
- CAPEX, OPEX, and debt service assumptions
- Equity and debt structure

### 6.2 Uncertainty Treatment

Rather than a single deterministic output, the model produces:
- P10 / P50 / P90 yearly revenues
- P10 / P50 / P90 yearly profits
- Distributions for NPV and IRR

These outputs allow users to assess **downside risk, base case, and upside potential**.

---

## 7. API Documentation

### 7.1 Endpoint Overview

```
POST /api/calc
```

### 7.2 Request Schema

```json
{
  "latitude": 51.0,
  "longitude": 9.0,
  "n_turbines": 5,
  "hub_height_m": 160,
  "turbine_type_id": 1,
  "equity_eur": 10000000,
  "tso_id": 2,
  "eeg_on": true,
  "cod_date": "2026-01-01",
  "fast_mode": true,
  "email": null
}
```

### 7.3 Response Schema (Excerpt)

```json
{
  "ok": true,
  "npv_eur": {"mean": 12000000, "p10": 2000000, "p90": 25000000},
  "irr": {"mean": 0.085, "p10": 0.03, "p90": 0.14},
  "yearly_table": [
    {
      "year": 2027,
      "revenue_p50": 3400000,
      "profit_p50": 1200000
    }
  ]
}
```

### 7.4 Error Handling

- HTTP 400: Invalid input or model failure
- Error payload includes traceback for debugging (development mode)

---

## 8. PDF Report Generation

In **precise mode**, the backend generates a **four-page PDF report** containing:

1. Executive summary and assumptions
2. Energy yield and seasonality
3. Prices, CAPEX, and OPEX
4. Financial results and risk indicators

Charts are generated programmatically and embedded directly into the report. Each report includes explicit disclaimers regarding model limitations.

---

## 9. Deployment

### 9.1 Frontend Deployment

The frontend can be deployed as a static application:

```bash
npm install
npm run build
```

The generated build artifacts can be served via:
- Nginx
- GitHub Pages
- Cloud hosting platforms (e.g., Vercel, Netlify)

### 9.2 Backend Deployment

Recommended deployment:
- Linux server or containerized environment
- Python virtual environment

```bash
pip install -r requirements.txt
set IS_LOCAL=1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Environment variables must be set for:
- Forecast data paths
- SMTP credentials (optional)

### 9.3 Scalability Considerations

- Backend is CPU-bound during forecasting
- Fast mode is recommended for public deployments
- Precise mode is suitable for queued or background execution

---

## 10. Assumptions and Limitations

- No micro-siting or wake interaction modeling
- Simplified representation of terrain and curtailment
- Market prices based on historical and aggregated data
- Financing structures are stylized

The application should be used only for **early-stage analysis**.

---

## 11. Disclaimer

This software produces **indicative, model-based estimates** only. It does not constitute investment advice, engineering advice, or financial recommendations. Any real-world project must be validated through site-specific measurements, engineering design, and professional due diligence.

---

## 12. Citation

If you use this software in academic work, please cite it as:

> *Wind Park Profitability and Risk Assessment Web Application, 2026.*

