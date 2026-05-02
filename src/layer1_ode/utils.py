"""
utils.py
--------
Funciones de utilidad para mecánica orbital:

- Conversión entre vectores de estado cartesianos y elementos orbitales
- Cálculo del semieje mayor desde posición y velocidad
- Energía orbital y verificación de conservación
- Utilidades de tiempo (JD ↔ fecha)
"""

import numpy as np
from astropy.time import Time
from .constants import GM_SOL_AU3_DAY2, KM_TO_AU


# ── ELEMENTOS ORBITALES ───────────────────────────────────────────────────

def state_to_orbital_elements(
    pos: np.ndarray,
    vel: np.ndarray,
    gm: float = GM_SOL_AU3_DAY2,
) -> dict:
    """
    Convierte un vector de estado cartesiano [pos, vel] a los
    elementos orbitales keplerianos clásicos.

    Args:
        pos : posición heliocéntrica [AU], shape (3,)
        vel : velocidad [AU/día], shape (3,)
        gm  : parámetro gravitacional [AU³/día²], default: Sol

    Returns:
        dict con:
            'a'     : semieje mayor [AU]
            'e'     : excentricidad
            'i'     : inclinación [grados]
            'omega' : argumento del perihelio [grados]
            'Omega' : longitud del nodo ascendente [grados]
            'M'     : anomalía media [grados]
            'E'     : energía orbital específica [AU²/día²]
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)

    # Energía orbital específica
    E = 0.5 * v**2 - gm / r

    # Semieje mayor: E = -GM / (2a) → a = -GM / (2E)
    if abs(E) < 1e-15:
        a = np.inf   # órbita parabólica
    else:
        a = -gm / (2.0 * E)

    # Vector de momento angular específico
    h_vec = np.cross(pos, vel)
    h = np.linalg.norm(h_vec)

    # Inclinación
    i = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))

    # Vector del nodo ascendente
    k = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k, h_vec)
    n = np.linalg.norm(n_vec)

    # Longitud del nodo ascendente
    if n < 1e-15:
        Omega = 0.0
    else:
        Omega = np.degrees(np.arccos(np.clip(n_vec[0] / n, -1, 1)))
        if n_vec[1] < 0:
            Omega = 360.0 - Omega

    # Vector de excentricidad (Laplace-Runge-Lenz)
    e_vec = (1.0/gm) * ((v**2 - gm/r) * pos - np.dot(pos, vel) * vel)
    e = np.linalg.norm(e_vec)

    # Argumento del perihelio
    if n < 1e-15 or e < 1e-10:
        omega = 0.0
    else:
        cos_omega = np.dot(n_vec, e_vec) / (n * e)
        omega = np.degrees(np.arccos(np.clip(cos_omega, -1, 1)))
        if e_vec[2] < 0:
            omega = 360.0 - omega

    # Anomalía verdadera
    if e < 1e-10:
        nu = 0.0
    else:
        cos_nu = np.dot(e_vec, pos) / (e * r)
        nu = np.degrees(np.arccos(np.clip(cos_nu, -1, 1)))
        if np.dot(pos, vel) < 0:
            nu = 360.0 - nu

    # Anomalía excéntrica y media
    if e < 1.0:
        cos_E = (e + np.cos(np.radians(nu))) / (1 + e * np.cos(np.radians(nu)))
        E_anom = np.degrees(np.arccos(np.clip(cos_E, -1, 1)))
        if nu > 180:
            E_anom = 360.0 - E_anom
        M = E_anom - np.degrees(e * np.sin(np.radians(E_anom)))
    else:
        M = 0.0

    return {
        "a"    : float(a),
        "e"    : float(e),
        "i"    : float(i),
        "omega": float(omega),
        "Omega": float(Omega),
        "M"    : float(M % 360.0),
        "E"    : float(E),
    }


def semi_major_axis(
    pos: np.ndarray,
    vel: np.ndarray,
    gm: float = GM_SOL_AU3_DAY2,
) -> float:
    """
    Calcula el semieje mayor oscular desde posición y velocidad.
    Función de conveniencia usada por la Capa 2 para calcular
    el semieje mayor en cada punto de la serie de residuos.

    Returns:
        a en AU
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    E = 0.5 * v**2 - gm / r
    return -gm / (2.0 * E) if abs(E) > 1e-15 else np.inf


# ── ENERGÍA Y CONSERVACIÓN ────────────────────────────────────────────────

def total_orbital_energy(
    pos: np.ndarray,
    vel: np.ndarray,
    gm: float = GM_SOL_AU3_DAY2,
) -> float:
    """
    Energía orbital específica del asteroide respecto al Sol.
    En un integrador ideal, este valor debe conservarse.
    Úsalo para diagnosticar si las tolerancias son suficientes.

    Returns:
        E en AU²/día²  (negativo = órbita elíptica)
    """
    return 0.5 * np.linalg.norm(vel)**2 - gm / np.linalg.norm(pos)


def check_energy_conservation(
    result: dict,
    tolerance_rel: float = 1e-6,
) -> dict:
    """
    Verifica la conservación de la energía orbital a lo largo de
    la integración. Una variación relativa > 1e-6 indica que las
    tolerancias del integrador deben ajustarse.

    Args:
        result        : salida de propagate() o propagate_from_state()
        tolerance_rel : variación relativa máxima aceptable

    Returns:
        dict con:
            'passed'        : bool
            'variation_rel' : variación relativa máxima (E_max - E_min) / |E_0|
            'E_initial'     : energía en t=0
            'E_final'       : energía en t=T
    """
    pos = result["asteroid_pos"]   # (N, 3)
    vel = result["asteroid_vel"]   # (N, 3)

    energies = np.array([
        total_orbital_energy(pos[i], vel[i])
        for i in range(len(pos))
    ])

    E0 = energies[0]
    variation = (energies.max() - energies.min()) / abs(E0) if abs(E0) > 1e-15 else 0.0
    passed = variation < tolerance_rel

    return {
        "passed"       : passed,
        "variation_rel": float(variation),
        "E_initial"    : float(E0),
        "E_final"      : float(energies[-1]),
    }


# ── CONVERSIONES DE TIEMPO ────────────────────────────────────────────────

def jd_to_iso(jd: float) -> str:
    """Convierte días julianos a fecha ISO 8601 (YYYY-MM-DD)."""
    return Time(jd, format="jd", scale="tdb").iso[:10]


def iso_to_jd(date_str: str) -> float:
    """Convierte fecha ISO 8601 a días julianos (TDB)."""
    return float(Time(date_str, format="iso", scale="tdb").jd)


def days_to_years(days: float) -> float:
    """Convierte días a años julianos (1 año = 365.25 días)."""
    return days / 365.25


# ── DISTANCIAS ────────────────────────────────────────────────────────────

def au_to_km(au: float) -> float:
    return au * 1.495978707e8


def au_to_ld(au: float) -> float:
    """Convierte AU a distancias lunares (1 LD = 384400 km)."""
    return au_to_km(au) / 384400.0
