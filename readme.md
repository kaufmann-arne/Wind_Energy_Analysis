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

This section describes how the web application estimates historical and future wind energy production at a user-selected location, and how the forecast is constructed and returned.

### 5.1 Data Sources

The forecast pipeline combines three main inputs:

- **ERA5 reanalysis data (ECMWF)**  
  Used to reconstruct historical wind conditions and derive climate features at the selected location. ERA5 is accessed as monthly NetCDF files.

- **Turbine-specific power curves**  
  Manufacturer power curves are used to translate hub-height wind speed into expected power/energy.

- **Pre-trained ML model artifacts**  
  A LightGBM-based correction-factor model trained on SCADA + ERA5 features. It predicts a **log correction factor** to adjust the physics-based energy estimate toward SCADA-realistic behavior. Artifacts include:
  - `final_model.pkl`
  - `feature_imputer.pkl`
  - `feature_cols.json`
  - `best_params.json`
  - `cv_metrics.csv`

**SCADA training data citations (used to train the correction model):**
- Plumley, C. & Takeuchi, R. (2025). *Kelmarsh wind farm data*. Zenodo. doi:10.5281/zenodo.16807551.  
- Plumley, C. & Takeuchi, R. (2025). *Penmanshiel wind farm data*. Zenodo. doi:10.5281/zenodo.16807304.  
- Byrne, R., & MacArtain, P. (2022). *Vestas V52 Wind Turbine, 10-minute SCADA Data, 2006–2020 - Dundalk Institute of Technology, Ireland*. Mendeley Data, V2. doi:10.17632/tm988rs48k.2.  
- Kaggle dataset: *3 x Vestas V100 2MW pitch power windspeed*. https://www.kaggle.com/datasets/morteneghj/3-x-vestas-v100-2mw-pitch-power-windspeed  

---

### 5.2 Forecast Methodology

For a given **location**, **hub height**, **turbine type**, **number of turbines**, and **Commissioning Date (COD)**, the application produces a **monthly energy production series** using a hybrid approach:

1. **Derive wind and climate features**  
   ERA5 variables are extracted for the selected location and transformed into the same feature set used during training (e.g., wind speed at relevant heights, direction from u/v, air density, roughness-related variables, seasonal/time features).

2. **Compute baseline (physics-based) expected energy**  
   Hub-height wind speed is mapped through the turbine power curve to obtain an expected power level. This provides a transparent baseline estimate.

3. **Apply the ML correction factor**  
   The ML model predicts a **log correction factor** for each time step / feature set. Energy is reconstructed as:

   E_corrected = E_baseline * exp(log_cf_hat)

   where:
   - E_baseline is the power-curve-based estimate,
   - log_cf_hat is the model-predicted log correction factor.

4. **Generate a long-horizon monthly forecast by resampling historical months**  
   To forecast future production at monthly resolution, the application uses a lightweight Monte Carlo resampling approach:
   - For each target calendar month (e.g., “January”), sample from historically observed/corrected January values.
   - Repeat for all months across the forecast horizon to form one production path.

5. **Align to COD and scale by turbine count**  
   - The forecast timeline is aligned so that month 1 corresponds to the COD month (or the next full month, depending on implementation).
   - Monthly energy is multiplied by the **number of turbines** .

---

### 5.3 Forecast Modes

Two runtime modes are supported to balance latency and statistical stability:

| Mode | Description |
|------|-------------|
| **Fast** | Reduced sampling / fewer Monte Carlo draws for a quick response suitable for interactive UI use. |
| **Precise** | Higher sampling resolution and typically a longer historical reference window (20 years of ERA5-derived monthly values) for improved stability and better representation of interannual variability. |

---

### 5.4 Output Definition

The forecast endpoint returns a **monthly time series** of energy production, including:

- `timestamp` (month start or ISO month identifier)
- `energy_kwh` (monthly total, scaled by turbine count)

### 5.5 Reproducibility: Building Datasets and Training the Model

This section documents the exact commands to reproduce the training pipeline used to generate the model artifacts consumed by the forecast service.

#### 5.5.1 Run the full pipeline (recommended order)

Run all commands from the repository root.

```bash
pip install -r requirements.txt
cd backend/wind-power-climate-ml
pip install -e .

# 1) Download ERA5 (uses config/sites.yaml by default)
python scripts/download_era5.py

# 2) Preprocess SCADA for all sites
#   - Download the SCADA datasets from the sources listed above.
#   - Place the raw files under: data/raw/Scada/Scada_<Site>.
#   - This step standardizes and aggregates the data and writes the final site-level CSVs to:
#     data/processed/Scada_final/
python scripts/run_scada_preprocess_sites.py

# 3) Build ML datasets -> data/mL/ (writes ml_dataset_*.csv and ml_dataset_ALL_T01.csv)
python scripts/build_ml_datasets.py

# 4) Validate the merged ML dataset (prints a validation report)
python scripts/validate_ml.py data/ml/ml_dataset_ALL_T01.csv --climate-prefix era5_

# 5) Train model: 
  python scripts/train_lgbm_louo.py \
  --data-path data/ml/ml_dataset_ALL_T01.csv \
  --out-dir artifacts/model_artifacts \
  --random-search \
  --n-iter 30 \
  --opt-metric rmse_energy_monthly_kwh
```
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

