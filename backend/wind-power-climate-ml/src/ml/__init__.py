"""ML subpackage.

The repository is intentionally lightweight and script-oriented. The main
modules are:

* :mod:`ml.dataset_builder` – build turbine-level ML datasets (SCADA + ERA5)
* :mod:`ml.validation` – sanity checks for the ML datasets
* :mod:`ml.training_lgbm` – LightGBM training and LOUO evaluation
* :mod:`ml.production_forecast` – website-time inference and monthly forecasting
"""

__all__ = [
    "dataset_builder",
    "validation",
    "training_lgbm",
    "production_forecast",
]
