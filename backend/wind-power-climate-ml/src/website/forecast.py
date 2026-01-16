"""Website production forecast API.

This module re-exports the website-facing forecast functions.
The implementation lives in `ml.production_forecast`.
"""

from ml.production_forecast import (
    build_hist_monthly_energy_lookup,
    forecast_monthly_20y_from_hist_energy,
)
