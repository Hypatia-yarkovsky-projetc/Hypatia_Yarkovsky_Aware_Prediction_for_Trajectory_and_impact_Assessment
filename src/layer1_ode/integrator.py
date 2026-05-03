"""
integrator.py
Integrador numérico del problema de N-cuerpos extendido con Yarkovsky.
Usa RK45 con tolerancias adaptativas y límite de paso para evitar
congelamientos en encuentros cercanos.
"""
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional
from .constants import GM_SOL_AU3_DAY2, DEFAULT_PERTURBERS, KM_TO_AU

def _compute_accelerations(y: np.ndarray, order: list[str], gm_map: dict, A2: float = 0.0) -> np.ndarray:
    """Calcula aceleraciones gravitacionales + Yarkovsky para todos los cuerpos."""
    n_bodies = len(order)
    acc = np.zeros((n_bodies, 3))
    
    # Extraer posiciones
    r = y[:n_bodies*3].reshape((n_bodies, 3))
    
    # Gravedad: N-cuerpos
    for i in range(1, n_bodies): # Asteroide y planetas
        for j in range(n_bodies):
            if i == j: continue
            r_ij = r[j] - r[i]
            dist = np.linalg.norm(r_ij)
            if dist < 1e-8: continue # Evitar singularidad
            acc[i] += gm_map[order[j]] * r_ij / (dist**3)
            
    # Yarkovsky (solo sobre el asteroide, índice 0)
    if A2 != 0.0:
        v_asteroid = y[n_bodies*3 : n_bodies*3 + 3]
        speed = np.linalg.norm(v_asteroid)
        if speed > 1e-10:
            # Aceleración tangencial unitaria
            t_hat = v_asteroid / speed
            # Escalamiento con distancia al Sol (r_0 = 1 AU)
            r_sun = np.linalg.norm(r[0])
            factor = A2 * (1.0 / r_sun)**2
            acc[0] += factor * t_hat
            
    return acc.flatten()

def _rhs(t: float, y: np.ndarray, order: list[str], gm_map: dict, A2: float) -> np.ndarray:
    """Función para solve_ivp: dy/dt = [v, a]."""
    n = len(order)
    v = y[n*3 : 2*n*3]
    a = _compute_accelerations(y, order, gm_map, A2)
    return np.concatenate([v, a])

def propagate_from_state(
    y0: np.ndarray,
    order: list[str],
    gm_map: dict,
    t_years: float,
    A2: float = 0.0,
    epoch_jd: float = 0.0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    max_step: float = 10.0, # NUEVO: Límite de paso en días
    verbose: bool = True,
) -> dict:
    """
    Propaga el sistema desde un estado inicial.
    
    Args:
        max_step : Paso máximo en DÍAS. Evita que el integrador haga saltos
                   gigantes en regiones de baja curvatura y se pierda, o que
                   se quede atascado reduciendo pasos infinitamente.
    """
    if verbose:
        print(f"[HYPATIA L1] Iniciando propagación: {t_years:.1f} años")
        print(f"[HYPATIA L1] A2 (Yarkovsky) = {A2:.2e} AU/día²")

    # solve_ivp espera tiempo en la misma unidad que las velocidades (días)
    t_span = (0.0, t_years * 365.25)
    
    # Evaluamos en puntos uniformes para obtener la trayectoria
    t_eval = np.linspace(0.0, t_span[1], max(100, int(t_years * 10)))

    try:
        sol = solve_ivp(
            fun=lambda t, y: _rhs(t, y, order, gm_map, A2),
            t_span=t_span,
            y0=y0,
            method='RK45',
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step, # <--- CRÍTICO
            dense_output=True,
        )
        
        if not sol.success:
            raise RuntimeError(f"Integración fallida: {sol.message}")
            
        # Extraer resultados
        n = len(order)
        times_jd = epoch_jd + sol.t
        
        # Reshape a (M, 3)
        pos_all = sol.y[:n*3, :].T.reshape((-1, n, 3))
        vel_all = sol.y[n*3:, :].T.reshape((-1, n, 3))
        
        asteroid_pos = pos_all[:, 0, :] # El asteroide siempre es el primero en 'order'
        asteroid_vel = vel_all[:, 0, :]
        
        if verbose:
            print(f"[HYPATIA L1] Propagación exitosa. {len(times_jd)} puntos.")
            
        return {
            "times_jd": times_jd,
            "asteroid_pos": asteroid_pos,
            "asteroid_vel": asteroid_vel,
            "success": True,
        }
        
    except Exception as e:
        print(f"[HYPATIA L1] Error crítico en integración: {e}")
        raise

def generate_uncertainty_cone(
    y0: np.ndarray,
    order: list[str],
    gm_map: dict,
    epoch_jd: float,
    dadt_mean: float,
    dadt_std: float,
    a_au: float,
    ecc: float,
    t_years: float,
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    """Genera n_samples de trayectorias variando da/dt según su distribución."""
    from .yarkovsky import dadt_to_A2
    
    rng = np.random.default_rng(seed)
    dadt_samples = rng.normal(dadt_mean, dadt_std, n_samples)
    
    trajectories = []
    for dadt in dadt_samples:
        A2 = dadt_to_A2(dadt, a_au, ecc)
        res = propagate_from_state(
            y0, order, gm_map, t_years, A2, epoch_jd, verbose=False
        )
        trajectories.append(res["asteroid_pos"])
        
    return {
        "times_jd": res["times_jd"],
        "trajectories": np.array(trajectories), # (n_samples, n_points, 3)
        "mean_traj": np.mean(trajectories, axis=0),
    }