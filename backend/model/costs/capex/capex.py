"""
capex.py

CAPEX calculator for wind park projects.

Design goals:
- Callable from a higher-level script (e.g., profit.py / power_to_profit.py)
- Input turbine type as an *index* (0, 1, 2) for compatibility with your UI/optimizer
- Return a dict that includes at least:
    - park_mw
    - total_capex_eur

Model summary:
- Turbine CAPEX is estimated via a regression ("HIK") using:
    - rated power (MW)
    - specific power density (SFL, W/m²) derived from rotor diameter
    - hub height (m)
- Balance of Plant (BoP) is added as a flat €/kW.

Notes:
- All costs are assumed to be in real EUR terms (not inflated).
- If you pass a horizon beyond the original calibration range, the regression still applies
  mathematically, but results may not be valid economically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


# -----------------------
# Constants
# -----------------------
KW_PER_MW: Final[float] = 1000.0

# BoP costs (e.g., foundations, grid connection, roads, planning)
# Source period in your comment: 2025–2028
BOP_EUR_PER_KW: Final[float] = 551.0


# -----------------------
# Turbine archetypes
# -----------------------
@dataclass(frozen=True)
class TurbineType:
    """Simple turbine archetype used for CAPEX estimation."""
    name: str
    p_mw: float
    rotor_diameter_m: float


# Catalog: feel free to extend/replace
TURBINE_TYPES_BY_KEY: dict[str, TurbineType] = {
    "LOW_WIND":  TurbineType("LOW_WIND", 5.56, 160.0),
    "BALANCED":  TurbineType("BALANCED", 5.7, 149.0),
    "HIGH_WIND": TurbineType("HIGH_WIND", 4.5, 136.0),
}

# Index mapping expected by profit.py
TURBINE_TYPE_INDEX_MAP: dict[int, str] = {
    0: "LOW_WIND",
    1: "BALANCED",
    2: "HIGH_WIND",
}


# -----------------------
# Engineering helpers
# -----------------------
def specific_power_density_w_per_m2(p_mw: float, rotor_diameter_m: float) -> float:
    """
    Specific rated power density (W/m²), based on rotor swept area.

    SFL (sometimes called "specific power") = rated_power / swept_area
    """
    if rotor_diameter_m <= 0:
        raise ValueError("rotor_diameter_m must be > 0")

    swept_area_m2 = math.pi * (rotor_diameter_m / 2.0) ** 2
    return (p_mw * 1_000_000.0) / swept_area_m2


def hik_eur_per_kw(p_mw: float, sfl_w_per_m2: float, hub_height_m: float) -> float:
    """
    HIK regression (EUR/kW) from German industry data (per your comment):

        HIK = 1476.19 - 65.62*P - 1.29*SFL + 3.50*NH

    Where:
      P   = rated power in MW
      SFL = specific power density in W/m²
      NH  = hub height in meters
    """
    return 1476.19 - 65.62 * p_mw - 1.29 * sfl_w_per_m2 + 3.50 * hub_height_m


# -----------------------
# Public API used by profit.py / power_to_profit.py
# -----------------------
def windpark_capex(
    *,
    n_turbines: int,
    turbine_type_id: int,
    hub_height_m: float,
    bop_eur_per_kw: float = BOP_EUR_PER_KW,
) -> dict[str, float | int | str]:
    """
    Compute wind park CAPEX from turbine archetype and hub height.

    Parameters
    ----------
    n_turbines:
        Number of turbines in the park (positive integer).
    turbine_type_id:
        Turbine type index:
            0 -> LOW_WIND
            1 -> BALANCED
            2 -> HIGH_WIND
    hub_height_m:
        Hub height in meters (> 0).
    bop_eur_per_kw:
        Balance of Plant cost in €/kW.

    Returns
    -------
    dict with keys including:
      - park_mw
      - total_capex_eur
    plus additional breakdown fields useful for debugging/reporting.
    """
    # ---- Validate inputs ----
    if not isinstance(n_turbines, int) or n_turbines <= 0:
        raise ValueError("n_turbines must be a positive integer")

    try:
        turbine_type_id = int(turbine_type_id)
    except Exception as e:
        raise ValueError("turbine_type_id must be an integer (0,1,2)") from e

    turbine_key = TURBINE_TYPE_INDEX_MAP.get(turbine_type_id)
    if turbine_key is None:
        raise ValueError(
            f"Unknown turbine_type_id={turbine_type_id}. "
            f"Valid: {sorted(TURBINE_TYPE_INDEX_MAP.keys())} "
            f"(0=LOW_WIND, 1=BALANCED, 2=HIGH_WIND)"
        )

    hub_height_m = float(hub_height_m)
    if hub_height_m <= 0:
        raise ValueError("hub_height_m must be > 0")

    bop_eur_per_kw = float(bop_eur_per_kw)
    if bop_eur_per_kw < 0:
        raise ValueError("bop_eur_per_kw must be >= 0")

    # ---- Resolve turbine archetype ----
    t = TURBINE_TYPES_BY_KEY[turbine_key]

    # ---- Derived technical values ----
    park_mw = float(n_turbines) * float(t.p_mw)
    sfl = specific_power_density_w_per_m2(t.p_mw, t.rotor_diameter_m)

    # ---- CAPEX calculation ----
    turbine_capex_eur_per_kw = hik_eur_per_kw(t.p_mw, sfl, hub_height_m)
    total_capex_eur_per_kw = turbine_capex_eur_per_kw + bop_eur_per_kw

    total_capex_eur = park_mw * KW_PER_MW * total_capex_eur_per_kw

    # Returning floats/ints/strings keeps it easy to JSON/log/print
    return {
        "turbine_type_id": int(turbine_type_id),
        "turbine_type": t.name,
        "n_turbines": int(n_turbines),
        "hub_height_m": float(hub_height_m),
        "park_mw": float(park_mw),
        "rotor_diameter_m": float(t.rotor_diameter_m),
        "sfl_w_per_m2": float(sfl),
        "turbine_capex_eur_per_kw": float(turbine_capex_eur_per_kw),
        "bop_eur_per_kw": float(bop_eur_per_kw),
        "total_capex_eur_per_kw": float(total_capex_eur_per_kw),
        "total_capex_eur": float(total_capex_eur),
    }


# -----------------------
# Convenience printing (optional)
# -----------------------
def print_windpark_result(n_turbines: int, turbine_type_id: int, hub_height_m: float) -> None:
    """Quick human-readable output for sanity checks."""
    res = windpark_capex(
        n_turbines=n_turbines,
        turbine_type_id=turbine_type_id,
        hub_height_m=hub_height_m,
    )

    print(
        f"Type={res['turbine_type']} (id={res['turbine_type_id']}) | "
        f"{res['n_turbines']} × {TURBINE_TYPES_BY_KEY[TURBINE_TYPE_INDEX_MAP[int(res['turbine_type_id'])]].p_mw} MW | "
        f"NH={res['hub_height_m']:.0f} m | "
        f"SFL={res['sfl_w_per_m2']:.0f} W/m² | "
        f"Turbine={res['turbine_capex_eur_per_kw']:.0f} €/kW | "
        f"BoP={res['bop_eur_per_kw']:.0f} €/kW | "
        f"Total={res['total_capex_eur_per_kw']:.0f} €/kW | "
        f"CAPEX={res['total_capex_eur'] / 1e6:.1f} M€"
    )


if __name__ == "__main__":
    print_windpark_result(12, 1, 160)  # BALANCED
    print_windpark_result(15, 2, 120)  # HIGH_WIND
    print_windpark_result(10, 0, 180)  # LOW_WIND
