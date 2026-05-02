"""
constants.py
------------
Constantes físicas y parámetros gravitacionales del sistema solar.

Todas las unidades están en el sistema AU / días / masas solares,
coherente con las efemérides de JPL Horizons.

Unidades:
    Distancia  : AU (unidad astronómica)
    Tiempo     : días
    Masa       : masas solares
    Velocidad  : AU/día
    Aceleración: AU/día²
"""

# ── Constante gravitacional en unidades AU³/(M_sol · día²) ────────────────
# Derivada de GM_sol = 1.32712440018e11 km³/s²
# Conversión: 1 AU = 1.495978707e8 km, 1 día = 86400 s
GM_SOL_AU3_DAY2 = 0.01720209895 ** 2   # (rad/día)² · AU³ → AU³/día²

# ── Parámetros gravitacionales μ = GM en AU³/día² ─────────────────────────
# Fuente: JPL DE440 planetary ephemeris constants
GM = {
    "sun"    : GM_SOL_AU3_DAY2,
    "mercury": 4.9125e-11,
    "venus"  : 7.2435e-10,
    "earth"  : 8.8877e-10,
    "moon"   : 1.0932e-11,
    "mars"   : 9.5495e-11,
    "jupiter": 2.8254e-07,
    "saturn" : 8.4597e-08,
    "uranus" : 1.2921e-08,
    "neptune": 1.5243e-08,
}

# IDs de JPL Horizons para cada cuerpo
JPL_IDS = {
    "sun"    : 10,
    "mercury": 199,
    "venus"  : 299,
    "earth"  : 399,
    "moon"   : 301,
    "mars"   : 499,
    "jupiter": 599,
    "saturn" : 699,
    "uranus" : 799,
    "neptune": 899,
}

# Cuerpos perturbadores activos por defecto (excluye luna para velocidad)
DEFAULT_PERTURBERS = ["sun", "venus", "earth", "mars", "jupiter", "saturn"]

# Distancia de referencia para el modelo de Yarkovsky (1 AU en AU)
R0_YARKOVSKY_AU = 1.0

# Conversión AU/My a AU/día (1 My = 365.25e6 días)
AU_MY_TO_AU_DAY = 1.0 / (365.25e6)

# Conversión km a AU
KM_TO_AU = 1.0 / 1.495978707e8
