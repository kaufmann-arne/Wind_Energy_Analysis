"""Website-facing utilities.

Thin wrappers around core ML production code with stable import paths.
"""

from .forecast import (
    build_hist_monthly_energy_lookup,
    forecast_monthly_20y_from_hist_energy,
)

__all__ = [
    "build_hist_monthly_energy_lookup",
    "forecast_monthly_20y_from_hist_energy",
]
