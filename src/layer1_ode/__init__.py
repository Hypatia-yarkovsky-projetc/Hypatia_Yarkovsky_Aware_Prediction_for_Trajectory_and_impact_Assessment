"""
layer1_ode
Capa 1 de HYPATIA: Motor de propagación orbital N-cuerpos + Yarkovsky.
Expone las funciones críticas para el pipeline maestro y las capas superiores.
"""

# Integrador y física
from .integrator import propagate_from_state, generate_uncertainty_cone
from .yarkovsky import (
    yarkovsky_acceleration,
    dadt_to_A2,
    A2_to_dadt,
    yarkovsky_order_of_magnitude,
)

# Condiciones iniciales y utilidades
from .initial_conditions import get_initial_conditions, pack_state_vector, unpack_state_vector
from .utils import (
    semi_major_axis,
    state_to_orbital_elements,
    au_to_km,
    jd_to_iso,
    iso_to_jd,
    check_energy_conservation,
)

# Validación y MOID
from .validation import fetch_ephemeris_arc, compare_scenarios, compute_position_errors
from .moid import cone_width_at_year

__all__ = [
    "propagate_from_state",
    "generate_uncertainty_cone",
    "yarkovsky_acceleration",
    "dadt_to_A2",
    "A2_to_dadt",
    "yarkovsky_order_of_magnitude",
    "get_initial_conditions",
    "pack_state_vector",
    "unpack_state_vector",
    "semi_major_axis",
    "state_to_orbital_elements",
    "au_to_km",
    "jd_to_iso",
    "iso_to_jd",
    "check_energy_conservation",
    "fetch_ephemeris_arc",
    "compare_scenarios",
    "compute_position_errors",
    "cone_width_at_year",
]