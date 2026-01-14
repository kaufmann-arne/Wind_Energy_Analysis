# capex.py
#
# Central callable CAPEX module for power_to_profit.py
# - Accepts turbine type as an INDEX (0,1,2) coming from profit.py
# - Accepts number of turbines and hub height
# - Returns a dict including keys used by power_to_profit.py:
#     - park_mw
#     - total_capex_eur
#
# Mapping (you can change ordering if you prefer):
#   0 -> LOW_WIND
#   1 -> BALANCED
#   2 -> HIGH_WIND

import math
from dataclasses import dataclass
from typing import Dict, Any


# -----------------------
# Turbine archetypes
# -----------------------
@dataclass(frozen=True)
class TurbineType:
    name: str
    p_mw: float
    rotor_diameter_m: float


TURBINE_TYPES_BY_KEY: Dict[str, TurbineType] = {
    "LOW_WIND":  TurbineType("LOW_WIND", 6.0, 170.0),
    "BALANCED":  TurbineType("BALANCED", 5.6, 160.0),
    "HIGH_WIND": TurbineType("HIGH_WIND", 4.2, 130.0),
}

# Index mapping expected by profit.py
TURBINE_TYPE_INDEX_MAP: Dict[int, str] = {
    0: "LOW_WIND",
    1: "BALANCED",
    2: "HIGH_WIND",
}

# -----------------------
# BoP costs (2025–2028)
# -----------------------
BOP_EUR_PER_KW = 551.0  # foundations, grid, roads, planning, etc.


# -----------------------
# Engineering functions
# -----------------------
def sfl_w_per_m2(p_mw: float, rotor_diameter_m: float) -> float:
    """Specific rated power density (W/m²) based on rotor swept area."""
    area = math.pi * (rotor_diameter_m / 2.0) ** 2
    return (p_mw * 1_000_000.0) / area


def hik_eur_per_kw(p_mw: float, sfl: float, hub_height_m: float) -> float:
    """
    HIK regression from German industry data:
    HIK = 1476.19 - 65.62*P - 1.29*SFL + 3.50*NH
    """
    return 1476.19 - 65.62 * p_mw - 1.29 * sfl + 3.50 * hub_height_m


# -----------------------
# Public API used by power_to_profit.py
# -----------------------
def windpark_capex(
    *,
    n_turbines: int,
    turbine_type_id: int,
    hub_height_m: float,
) -> Dict[str, Any]:
    """
    Compute wind park CAPEX based on turbine archetype + hub height.

    Parameters
    ----------
    n_turbines : int
        Number of turbines in the park.
    turbine_type_id : int
        Turbine type index coming from profit.py:
          0 -> LOW_WIND
          1 -> BALANCED
          2 -> HIGH_WIND
    hub_height_m : float
        Hub height in meters.

    Returns
    -------
    dict including:
      - park_mw
      - total_capex_eur
      plus breakdown fields.
    """
    if not isinstance(n_turbines, int) or n_turbines <= 0:
        raise ValueError("n_turbines must be a positive integer.")

    try:
        turbine_type_id = int(turbine_type_id)
    except Exception as e:
        raise ValueError("turbine_type_id must be an integer (0,1,2).") from e

    if turbine_type_id not in TURBINE_TYPE_INDEX_MAP:
        raise ValueError(
            f"Unknown turbine_type_id={turbine_type_id}. "
            f"Valid: {sorted(TURBINE_TYPE_INDEX_MAP.keys())} "
            f"(0=LOW_WIND, 1=BALANCED, 2=HIGH_WIND)"
        )

    hub_height_m = float(hub_height_m)
    if hub_height_m <= 0:
        raise ValueError("hub_height_m must be > 0.")

    turbine_key = TURBINE_TYPE_INDEX_MAP[turbine_type_id]
    t = TURBINE_TYPES_BY_KEY[turbine_key]

    park_mw = float(n_turbines) * float(t.p_mw)
    sfl = sfl_w_per_m2(t.p_mw, t.rotor_diameter_m)

    turbine_capex_eur_per_kw = hik_eur_per_kw(t.p_mw, sfl, hub_height_m)
    total_capex_eur_per_kw = float(turbine_capex_eur_per_kw) + float(BOP_EUR_PER_KW)

    total_capex_eur = park_mw * 1000.0 * total_capex_eur_per_kw

    return {
        "turbine_type_id": int(turbine_type_id),
        "turbine_type": t.name,
        "n_turbines": int(n_turbines),
        "hub_height_m": float(hub_height_m),
        "park_mw": float(park_mw),
        "rotor_diameter_m": float(t.rotor_diameter_m),
        "sfl": float(sfl),
        "turbine_capex_eur_per_kw": float(turbine_capex_eur_per_kw),
        "bop_eur_per_kw": float(BOP_EUR_PER_KW),
        "total_capex_eur_per_kw": float(total_capex_eur_per_kw),
        "total_capex_eur": float(total_capex_eur),
    }


# -----------------------
# Convenience helper (optional)
# -----------------------
def print_windpark_result(n_turbines: int, turbine_type_id: int, hub_height_m: float) -> None:
    res = windpark_capex(
        n_turbines=n_turbines,
        turbine_type_id=turbine_type_id,
        hub_height_m=hub_height_m,
    )
    print(
        f"Type={res['turbine_type']} (id={res['turbine_type_id']}) | "
        f"{res['n_turbines']} × {TURBINE_TYPES_BY_KEY[res['turbine_type']].p_mw} MW | "
        f"NH={res['hub_height_m']:.0f} m | "
        f"SFL={res['sfl']:.0f} W/m² | "
        f"Turbine={res['turbine_capex_eur_per_kw']:.0f} €/kW | "
        f"BoP={res['bop_eur_per_kw']:.0f} €/kW | "
        f"Total={res['total_capex_eur_per_kw']:.0f} €/kW | "
        f"CAPEX={res['total_capex_eur']/1e6:.1f} M€"
    )


# -----------------------
# Example
# -----------------------
if __name__ == "__main__":
    print_windpark_result(12, 1, 160)  # BALANCED
    print_windpark_result(15, 2, 120)  # HIGH_WIND
    print_windpark_result(10, 0, 180)  # LOW_WIND
