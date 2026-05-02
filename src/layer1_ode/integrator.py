"""
integrator.py (v2.1 - Optimizado)
Núcleo RK45 N-cuerpos + Yarkovsky. Vectorizado y estable para encuentros cercanos.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .constants import DEFAULT_PERTURBERS, RTOL, ATOL, MAX_STEP_DAYS
from .initial_conditions import get_initial_conditions, pack_state_vector
from .yarkovsky import dadt_to_A2, yarkovsky_acceleration

def _build_rhs(order: list[str], gm_map: dict, A2: float):
    """
    RHS vectorizado y numéricamente estable para N-cuerpos + Yarkovsky.
    """
    n = len(order)
    gm = np.array([gm_map[name] for name in order])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        pos = y[:3*n].reshape(n, 3)
        vel = y[3*n:].reshape(n, 3)
        acc = np.zeros((n, 3))

        # ── Gravedad vectorizada segura ─────────────────────────────
        r_ij = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        dist = np.linalg.norm(r_ij, axis=2)

        # Protección contra división por cero y auto-gravedad
        np.fill_diagonal(dist, np.inf)
        dist = np.where(dist < 1e-10, np.inf, dist)  # ← CRÍTICO

        inv_dist3 = 1.0 / dist**3
        acc += np.sum(gm[np.newaxis, :, np.newaxis] * r_ij * inv_dist3[:, :, np.newaxis], axis=1)

        # ── Yarkovsky (solo asteroide, índice 0) ────────────────────
        if A2 != 0.0:
            acc[0] += yarkovsky_acceleration(pos[0], vel[0], A2)

        return np.concatenate([vel.ravel(), acc.ravel()])
    return rhs

def _run_solver(rhs, y0, t_span, t_eval, dense_output):
    return solve_ivp(
        rhs, t_span, y0, method="RK45", t_eval=t_eval,
        rtol=1e-9, atol=1e-12, dense_output=dense_output,
        max_step=5.0,  # ← Evita saltos en encuentros cercanos
    )

def propagate(asteroid_id, epoch_start, t_years, dadt_au_my=0.0, 
              a_au=1.0, ecc=0.2, perturbers=DEFAULT_PERTURBERS, dense_output=True):
    A2 = dadt_to_A2(dadt_au_my, a_au, ecc) if dadt_au_my != 0.0 else 0.0
    ic = get_initial_conditions(asteroid_id, epoch_start, perturbers)
    y0, order, gm_map = pack_state_vector(ic)
    epoch_jd = ic["epoch_jd"]

    t_span = (0.0, t_years * 365.25)
    # Muestreo diario para salida, pero el integrador usa pasos adaptativos internos
    n_steps = max(int(abs(t_years) * 365.25), 100)
    t_eval = np.linspace(*t_span, n_steps)

    print(f"[HYPATIA] Integrando {t_years:.1f} años | A2={A2:.3e} | Pasos adaptativos activos...")
    
    rhs = _build_rhs(order, gm_map, A2)
    sol = _run_solver(rhs, y0, t_span, t_eval, dense_output)
    if not sol.success:
        raise RuntimeError(f"Integrador no convergió: {sol.message}")

    n_bodies = len(order)
    pos_all = sol.y[:3*n_bodies, :].reshape(n_bodies, 3, -1)
    vel_all = sol.y[3*n_bodies:, :].reshape(n_bodies, 3, -1)

    return {
        "sol": sol, "times_jd": epoch_jd + sol.t, "times_days": sol.t,
        "asteroid_pos": pos_all[0].T, "asteroid_vel": vel_all[0].T,
        "order": order, "A2": A2, "epoch_jd": epoch_jd, "n_bodies": n_bodies
    }

def propagate_from_state(y0, order, gm_map, t_years, A2=0.0, epoch_jd=0.0, dense_output=True):
    t_span = (0.0, t_years * 365.25)
    n_steps = max(int(abs(t_years) * 365.25), 100)
    t_eval = np.linspace(*t_span, n_steps)

    rhs = _build_rhs(order, gm_map, A2)
    sol = _run_solver(rhs, y0, t_span, t_eval, dense_output)
    if not sol.success:
        raise RuntimeError(f"Integrador no convergió: {sol.message}")

    n_bodies = len(order)
    pos_all = sol.y[:3*n_bodies, :].reshape(n_bodies, 3, -1)
    vel_all = sol.y[3*n_bodies:, :].reshape(n_bodies, 3, -1)

    return {
        "sol": sol, "times_jd": epoch_jd + sol.t, "times_days": sol.t,
        "asteroid_pos": pos_all[0].T, "asteroid_vel": vel_all[0].T,
        "order": order, "A2": A2, "epoch_jd": epoch_jd, "n_bodies": n_bodies
    }