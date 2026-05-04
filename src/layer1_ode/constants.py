"""
constants.py
Constantes físicas y parámetros gravitacionales del sistema solar.
Sistema de unidades: AU / días / AU³/día² (coherente con JPL Horizons)
"""
# Constante gravitacional solar (constante gaussiana k²)
GM_SOL_AU3_DAY2 = 0.01720209895 ** 2  # ≈ 2.959122083e-04 AU³/día²

# Parámetros gravitacionales μ = GM en AU³/día² (DE440)
GM = {
    "sun": GM_SOL_AU3_DAY2,
    "mercury": 4.9125e-11,
    "venus": 7.2435e-10,
    "earth": 8.8877e-10,
    "moon": 1.0932e-11,
    "mars": 9.5495e-11,
    "jupiter": 2.8254e-07,
    "saturn": 8.4597e-08,
    "uranus": 1.2921e-08,
    "neptune": 1.5243e-08,
}

JPL_IDS = {
    "sun": 10, "mercury": 199, "venus": 299, "earth": 399,
    "moon": 301, "mars": 499, "jupiter": 599, "saturn": 699,
    "uranus": 799, "neptune": 899,
}

DEFAULT_PERTURBERS = ["sun", "venus", "earth", "mars", "jupiter", "saturn"]
R0_YARKOVSKY_AU = 1.0
AU_MY_TO_AU_DAY = 1.0 / (365.25e6)
KM_PER_AU = 1.495978707e8
KM_TO_AU = 1.0 / KM_PER_AU

# Tolerancias oficiales del proyecto
RTOL = 1e-9
ATOL = 1e-12
MAX_STEP_DAYS = 5.0  # Crítico: evita saltar encuentros cercanos o periapsis