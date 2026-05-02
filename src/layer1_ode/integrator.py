"""
integrator.py
-------------
Núcleo de la Capa 1 de HYPATIA: integrador numérico del problema de
N cuerpos gravitacional extendido con la perturbación de Yarkovsky.

Sistema de EDOs:
    dy/dt = f(t, y)

donde y = [r_ast, v_ast, r_p1, v_p1, ..., r_pN, v_pN]
es el vector de estado concatenado (6 componentes por cuerpo).

La ecuación de movimiento del asteroide incluye el término de Yarkovsky:
    d²r_ast/dt² = Σⱼ GMⱼ·(rⱼ−r_ast)/|rⱼ−r_ast|³  +  A2·(r0/r)²·t̂

Los planetas perturbadores siguen la mecánica gravitacional pura
(sin Yarkovsky, ya que la perturbación sobre ellos es despreciable).

Método de integración: RK45 (Dormand-Prince) con paso adaptativo.
    rtol = 1e-9 (tolerancia relativa)
    atol = 1e-12 (tolerancia absoluta)
"""

import numpy as np
from scipy.integrate import solve_ivp
from astropy.time import Time

from .constants import DEFAULT_PERTURBERS, KM_TO_AU
from .initial_conditions import get_initial_conditions, pack_state_vector, unpack_state_vector
from .yarkovsky import yarkovsky_acceleration, dadt_to_A2


# ── Tolerancias del integrador ────────────────────────────────────────────
RTOL = 1e-9
ATOL = 1e-12


def _build_rhs(order: list[str], gm_map: dict, A2: float):
    """
    Construye la función del lado derecho (RHS) del sistema de EDOs.

    Retorna una función f(t, y) compatible con scipy.integrate.solve_ivp.

    Args:
        order  : lista de nombres de cuerpos (primer elemento: 'asteroid')
        gm_map : dict {nombre: GM en AU³/día²}
        A2     : parámetro de Yarkovsky del asteroide en AU/día²
    """
    n_bodies = len(order)
    ast_idx = 0      # el asteroide siempre es el primer cuerpo

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        # Desempaquetar posiciones y velocidades
        pos = y[: 3 * n_bodies].reshape(n_bodies, 3)
        vel = y[3 * n_bodies :].reshape(n_bodies, 3)
        acc = np.zeros((n_bodies, 3))

        # ── Aceleración gravitacional mutua ───────────────────────────
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i == j:
                    continue
                if gm_map[order[j]] == 0.0:
                    continue   # cuerpos sin masa (no contribuyen)
                r_ij = pos[j] - pos[i]
                dist = np.linalg.norm(r_ij)
                if dist < 1e-10:
                    continue   # protección numérica (cuerpos coincidentes)
                acc[i] += gm_map[order[j]] * r_ij / dist ** 3

        # ── Perturbación de Yarkovsky (solo sobre el asteroide) ───────
        if A2 != 0.0:
            acc[ast_idx] += yarkovsky_acceleration(
                pos[ast_idx], vel[ast_idx], A2
            )

        # Ensamblar derivada: dy/dt = [vel, acc]
        return np.concatenate([vel.flatten(), acc.flatten()])

    return rhs


def propagate(
    asteroid_id: int | str,
    epoch_start: str | float,
    t_years: float,
    dadt_au_my: float = 0.0,
    a_au: float = 1.0,
    ecc: float = 0.2,
    perturbers: list[str] = DEFAULT_PERTURBERS,
    dense_output: bool = False,
) -> dict:
    """
    Propaga la órbita del asteroide desde epoch_start durante t_years años.

    Args:
        asteroid_id  : ID JPL del asteroide (ej. 99942 para Apophis)
        epoch_start  : Época inicial ('YYYY-MM-DD' o JD float)
        t_years      : Duración de la integración en años (positivo: futuro)
        dadt_au_my   : Parámetro de Yarkovsky en AU/My
                       0.0 = modelo gravitacional puro (sin Yarkovsky)
        a_au         : Semieje mayor del asteroide en AU (para convertir A2)
        ecc          : Excentricidad del asteroide (para convertir A2)
        perturbers   : Cuerpos perturbadores a incluir
        dense_output : Si True, guarda la solución continua interpolable

    Returns:
        dict con claves:
            'sol'        : objeto OdeResult de scipy
            'times_jd'   : np.ndarray, tiempos en JD
            'times_days' : np.ndarray, tiempos en días desde el inicio
            'asteroid_pos': np.ndarray (N, 3), posiciones en AU
            'asteroid_vel': np.ndarray (N, 3), velocidades en AU/día
            'order'      : lista de nombres de cuerpos
            'A2'         : float, parámetro de Yarkovsky usado
            'epoch_jd'   : float, época inicial en JD
    """
    # Convertir da/dt → A2
    A2 = dadt_to_A2(dadt_au_my, a_au, ecc) if dadt_au_my != 0.0 else 0.0

    # Obtener condiciones iniciales desde JPL Horizons
    ic = get_initial_conditions(asteroid_id, epoch_start, perturbers)
    y0, order, gm_map = pack_state_vector(ic)
    epoch_jd = ic["epoch_jd"]

    # Intervalo de integración en días
    t_span = (0.0, t_years * 365.25)
    # Puntos de evaluación: cada 30 días aproximadamente
    n_steps = max(int(abs(t_years) * 12), 2)
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    print(f"[HYPATIA] Integrando {t_years:.1f} años "
          f"| A2={A2:.3e} AU/día² "
          f"| da/dt={dadt_au_my:.4f} AU/My")

    # Construir RHS y lanzar integrador
    rhs = _build_rhs(order, gm_map, A2)

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=RTOL,
        atol=ATOL,
        dense_output=dense_output,
    )

    if not sol.success:
        raise RuntimeError(f"Integrador no convergió: {sol.message}")

    n_bodies = len(order)
    ast_idx = 0

    # Extraer trayectoria del asteroide
    pos_all = sol.y[: 3 * n_bodies, :].reshape(n_bodies, 3, -1)
    vel_all = sol.y[3 * n_bodies :, :].reshape(n_bodies, 3, -1)

    asteroid_pos = pos_all[ast_idx].T   # shape (N, 3)
    asteroid_vel = vel_all[ast_idx].T   # shape (N, 3)

    times_jd = epoch_jd + sol.t

    print(f"[HYPATIA] Integración completada: {len(sol.t)} pasos evaluados.")

    return {
        "sol"         : sol,
        "times_jd"    : times_jd,
        "times_days"  : sol.t,
        "asteroid_pos": asteroid_pos,
        "asteroid_vel": asteroid_vel,
        "order"       : order,
        "A2"          : A2,
        "epoch_jd"    : epoch_jd,
        "n_bodies"    : n_bodies,
    }


def propagate_from_state(
    y0: np.ndarray,
    order: list[str],
    gm_map: dict,
    t_years: float,
    A2: float = 0.0,
    epoch_jd: float = 0.0,
) -> dict:
    """
    Versión baja-latencia de propagate() que acepta directamente
    un vector de estado y0 pre-descargado. Útil para el pipeline
    interno (la Capa 2 llama a esto repetidamente con distintos A2).

    Args:
        y0      : vector de estado inicial empaquetado (de pack_state_vector)
        order   : lista de nombres de cuerpos
        gm_map  : dict {nombre: GM}
        t_years : años a integrar
        A2      : parámetro de Yarkovsky en AU/día²
        epoch_jd: época de inicio en JD (para calcular times_jd)

    Returns:
        Mismo formato que propagate()
    """
    t_span = (0.0, t_years * 365.25)
    n_steps = max(int(abs(t_years) * 12), 2)
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    rhs = _build_rhs(order, gm_map, A2)

    sol = solve_ivp(
        rhs, t_span, y0,
        method="RK45", t_eval=t_eval,
        rtol=RTOL, atol=ATOL,
    )

    if not sol.success:
        raise RuntimeError(f"Integrador no convergió: {sol.message}")

    n_bodies = len(order)
    pos_all = sol.y[: 3 * n_bodies, :].reshape(n_bodies, 3, -1)
    vel_all = sol.y[3 * n_bodies :, :].reshape(n_bodies, 3, -1)

    return {
        "sol"         : sol,
        "times_jd"    : epoch_jd + sol.t,
        "times_days"  : sol.t,
        "asteroid_pos": pos_all[0].T,
        "asteroid_vel": vel_all[0].T,
        "order"       : order,
        "A2"          : A2,
        "epoch_jd"    : epoch_jd,
        "n_bodies"    : n_bodies,
    }
