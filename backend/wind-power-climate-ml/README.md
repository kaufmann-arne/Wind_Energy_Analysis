# Wind Power + ERA5 + SCADA ML Pipeline

This repository contains a reproducible pipeline to:

1. Download ERA5 reanalysis (hourly) for sites / bounding boxes.
2. Preprocess heterogeneous SCADA exports into a unified hourly dataset.
3. Merge SCADA with ERA5, engineer climate features, and build ML-ready datasets.
4. Validate dataset integrity (sanity checks).
5. Train a LightGBM correction-factor model (Leave-One-Turbine-Out CV).
6. Run production-style inference to build historical monthly energy and
   generate 20-year monthly forecasts (Monte Carlo resampling by calendar month).

The code is split into importable modules under `src/` and runnable entrypoints under `scripts/`.

## Repository layout

- `src/era5/` ERA5 download helpers (CDS API).
- `src/scada/` SCADA parsing, availability logic, hourly/park aggregation.
- `src/ml/` ERA5+SCADA merge, feature engineering, dataset building, validation.
- `src/ml/training_lgbm.py` LGBM training utilities (explicit features; LOUO).
- `src/ml/production_forecast.py` Website inference + long-horizon monthly forecast.
- `scripts/` CLI entrypoints.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Notes

- ERA5 download requires a configured CDS API key (`~/.cdsapirc`).
- The pipeline uses UTC timestamps end-to-end.

## Training (LightGBM)

```bash
python scripts/train_lgbm_louo.py \
  --data-path Data/ML \
  --out-dir artifacts/model_artifacts \
  --do-energy-eval
```

Outputs:
* `final_model.pkl`
* `feature_imputer.pkl`
* `feature_cols.json`
* `cv_metrics.csv`

## Production forecast (monthly)

```bash
python scripts/forecast_monthly_energy.py \
  --lat 52.52 --lon 13.41 --hub-height-m 75 \
  --turbine-type Senvion_MM82 \
  --era5-main-dir Data/Raw/Era5/ERA5_Germany_ClimateStore \
  --era5-d2m-dir  Data/Raw/Era5/ERA5_Germany_d2m \
  --turbine-power-curve-json Data/turbine_power_curves.json \
  --model-dir artifacts/model_artifacts \
  --out-dir out_forecasts
```


## Local path setup (your project structure)

Your current project directories are:

- Model artifacts:
  - `LGBM_Model/model_artifacts/feature_imputer.pkl`
  - `LGBM_Model/model_artifacts/final_model.pkl`
  - `LGBM_Model/model_artifacts/cv_metrics.csv`
  - `LGBM_Model/model_artifacts/feature_cols.json`
- Raw ERA5: `Data/Raw/Era5/`
- Raw SCADA: `Data/Raw/Scada/`

If you want to keep machine-specific absolute paths, copy the example config and adjust it:

- `config/paths.windows.johan.yaml` (example; uses Windows absolute paths)

Recommended approach for GitHub:

1. Copy it to `config/paths.local.yaml` and adjust as needed.
2. Keep `config/paths.local.yaml` untracked (it is ignored by `.gitignore`).

The CLI scripts do not require a config file (they accept explicit `--era5-main-dir`, `--model-dir`, etc.), but keeping a local config is helpful when you run the pipeline repeatedly.
