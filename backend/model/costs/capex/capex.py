import math
from dataclasses import dataclass

# -----------------------
# Turbine archetypes
# -----------------------
@dataclass(frozen=True)
class TurbineType:
    name: str
    p_mw: float
    rotor_diameter_m: float

TURBINE_TYPES = {
    "LOW_WIND":  TurbineType("LOW_WIND", 6.0, 170.0),
    "BALANCED":  TurbineType("BALANCED", 5.6, 160.0),
    "HIGH_WIND": TurbineType("HIGH_WIND", 4.2, 130.0),
}

# -----------------------
# BoP costs (2025–2028)
# -----------------------
BOP_EUR_PER_KW = 551   # foundations, grid, roads, planning, etc.

# -----------------------
# Engineering functions
# -----------------------
def sfl_w_per_m2(p_mw, rotor_diameter_m):
    area = math.pi * (rotor_diameter_m / 2) ** 2
    return (p_mw * 1_000_000) / area

def hik_eur_per_kw(p_mw, sfl, hub_height_m):
    """
    HIK regression from German industry data:
    HIK = 1476.19 - 65.62*P - 1.29*SFL + 3.50*NH
    """
    return 1476.19 - 65.62 * p_mw - 1.29 * sfl + 3.50 * hub_height_m

# -----------------------
# Wind park CAPEX
# -----------------------
def windpark_capex(n_turbines, turbine_type_key, hub_height_m):
    t = TURBINE_TYPES[turbine_type_key]

    park_mw = n_turbines * t.p_mw
    sfl = sfl_w_per_m2(t.p_mw, t.rotor_diameter_m)

    turbine_capex_kw = hik_eur_per_kw(t.p_mw, sfl, hub_height_m)
    total_capex_kw = turbine_capex_kw + BOP_EUR_PER_KW

    total_capex = park_mw * 1000 * total_capex_kw

    return {
        "park_mw": park_mw,
        "sfl": sfl,
        "turbine_capex_eur_per_kw": turbine_capex_kw,
        "bop_eur_per_kw": BOP_EUR_PER_KW,
        "total_capex_eur_per_kw": total_capex_kw,
        "total_capex_eur": total_capex
    }

# -----------------------
# Pretty printer
# -----------------------
def print_windpark_result(n_turbines, turbine_type, hub_height_m):
    res = windpark_capex(n_turbines, turbine_type, hub_height_m)
    t = TURBINE_TYPES[turbine_type]

    print(
        f"{turbine_type} ({n_turbines} × {t.p_mw} MW) | "
        f"NH={hub_height_m} m | "
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
    print_windpark_result(12, "BALANCED", 160)
    print_windpark_result(15, "HIGH_WIND", 120)
    print_windpark_result(10, "LOW_WIND", 180)
