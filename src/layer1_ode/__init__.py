"""
layer1_ode
Capa 1 de HYPATIA: Integrador N-cuerpos con perturbación de Yarkovsky.
API pública del módulo:
    from src.layer1_ode import propagate, run_validation, generate_uncertainty_cone
"""
from .integrator import propagate, propagate_from_state
from .initial_conditions import (
    get_initial_conditions,
    pack_state_vector,
    unpack_state_vector,
)
from .yarkovsky import (
    yarkovsky_acceleration,
    dadt_to_A2,
    A2_to_dadt,
    yarkovsky_order_of_magnitude,
)
from .moid import (
    compute_moid_timeseries,
    find_close_approaches,
    generate_uncertainty_cone,
    cone_width_at_year,
)
from .validation import (
    run_validation,
    compare_scenarios,
    fetch_ephemeris_arc,
    compute_position_errors,
)
from .utils import (
    state_to_orbital_elements,
    semi_major_axis,
    check_energy_conservation,
    jd_to_iso,
    iso_to_jd,
    au_to_km,
    au_to_ld,
)
from .constants import GM, JPL_IDS, DEFAULT_PERTURBERS, RTOL, ATOL, MAX_STEP_DAYS

__all__ = [
    "propagate", "propagate_from_state",
    "get_initial_conditions", "pack_state_vector", "unpack_state_vector",
    "yarkovsky_acceleration", "dadt_to_A2", "A2_to_dadt", "yarkovsky_order_of_magnitude",
    "compute_moid_timeseries", "find_close_approaches", "generate_uncertainty_cone", "cone_width_at_year",
    "run_validation", "compare_scenarios", "fetch_ephemeris_arc", "compute_position_errors",
    "state_to_orbital_elements", "semi_major_axis", "check_energy_conservation",
    "jd_to_iso", "iso_to_jd", "au_to_km", "au_to_ld",
    "GM", "JPL_IDS", "DEFAULT_PERTURBERS", "RTOL", "ATOL", "MAX_STEP_DAYS"
]