"""
layer1_ode
----------
Capa 1 de HYPATIA: Integrador N-cuerpos con perturbación de Yarkovsky.

API pública del módulo:

    from src.layer1_ode import propagate, run_validation, generate_uncertainty_cone

Funciones principales:
    propagate()                 → Propaga la órbita del asteroide
    propagate_from_state()      → Versión rápida para uso interno (pipeline)
    run_validation()            → Valida el integrador contra JPL
    compare_scenarios()         → Compara escenarios con/sin Yarkovsky
    generate_uncertainty_cone() → Genera el cono de incertidumbre orbital
    compute_moid_timeseries()   → Calcula el MOID en ventanas temporales
    find_close_approaches()     → Filtra acercamientos peligrosos
    get_initial_conditions()    → Descarga condiciones iniciales de JPL
    state_to_orbital_elements() → Convierte estado cartesiano a elementos
    dadt_to_A2()                → Convierte da/dt → parámetro A2
    A2_to_dadt()                → Convierte A2 → da/dt
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
from .constants import GM, JPL_IDS, DEFAULT_PERTURBERS

__all__ = [
    # Integrador
    "propagate",
    "propagate_from_state",
    # Condiciones iniciales
    "get_initial_conditions",
    "pack_state_vector",
    "unpack_state_vector",
    # Yarkovsky
    "yarkovsky_acceleration",
    "dadt_to_A2",
    "A2_to_dadt",
    "yarkovsky_order_of_magnitude",
    # MOID y cono
    "compute_moid_timeseries",
    "find_close_approaches",
    "generate_uncertainty_cone",
    "cone_width_at_year",
    # Validación
    "run_validation",
    "compare_scenarios",
    "fetch_ephemeris_arc",
    "compute_position_errors",
    # Utilidades
    "state_to_orbital_elements",
    "semi_major_axis",
    "check_energy_conservation",
    "jd_to_iso",
    "iso_to_jd",
    "au_to_km",
    "au_to_ld",
    # Constantes
    "GM",
    "JPL_IDS",
    "DEFAULT_PERTURBERS",
]
