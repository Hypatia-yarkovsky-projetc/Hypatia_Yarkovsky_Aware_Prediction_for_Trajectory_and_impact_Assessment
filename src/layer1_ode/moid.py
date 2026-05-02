"""
moid.py
-------
Cálculo de la Minimum Orbit Intersection Distance (MOID) y
generación del cono de incertidumbre orbital.

MOID: distancia mínima entre la trayectoria del asteroide y la de
la Tierra en una ventana temporal dada. Es la métrica directa de
proximidad orbital y el primer indicador de riesgo de impacto.

Cono de incertidumbre: conjunto de trayectorias generadas al variar
da/dt dentro del intervalo de confianza estimado por la Capa 2.
"""

import numpy as np
from typing import Generator
from .integrator import propagate_from_state
from .yarkovsky import dadt_to_A2


# ── MOID ─────────────────────────────────────────────────────────────────

def compute_moid_timeseries(
    result_asteroid: dict,
    result_earth: dict,
    window_years: float = 5.0,
) -> list[dict]:
    """
    Calcula el MOID en ventanas temporales solapadas a lo largo de
    la trayectoria integrada.

    Args:
        result_asteroid : dict de salida de propagate() para el asteroide
        result_earth    : dict de salida de propagate() para la Tierra
                          (misma época y duración)
        window_years    : ancho de cada ventana en años

    Returns:
        Lista de dicts con claves:
            't_center_jd' : JD central de la ventana
            'moid_au'     : MOID en AU dentro de la ventana
            'moid_km'     : MOID en km
            'idx_min'     : índice del paso con distancia mínima
    """
    times_ast = result_asteroid["times_jd"]
    pos_ast   = result_asteroid["asteroid_pos"]   # (N, 3)
    pos_ear   = result_earth["asteroid_pos"]      # (N, 3) — Earth pos

    # Distancias en cada paso de tiempo
    diff = pos_ast - pos_ear
    dist_au = np.linalg.norm(diff, axis=1)        # (N,)

    # Construir ventanas
    window_days = window_years * 365.25
    t_start = times_ast[0]
    t_end   = times_ast[-1]
    results = []

    t_win = t_start
    while t_win < t_end:
        mask = (times_ast >= t_win) & (times_ast < t_win + window_days)
        if mask.sum() < 2:
            t_win += window_days
            continue

        dist_window = dist_au[mask]
        idx_local   = np.argmin(dist_window)
        idx_global  = np.where(mask)[0][idx_local]

        moid_au = dist_window[idx_local]
        results.append({
            "t_center_jd": times_ast[mask][idx_local],
            "moid_au"    : float(moid_au),
            "moid_km"    : float(moid_au / 6.685e-9),   # AU → km
            "idx_min"    : int(idx_global),
        })
        t_win += window_days

    return results


def find_close_approaches(
    moid_series: list[dict],
    threshold_au: float = 0.05,
) -> list[dict]:
    """
    Filtra los acercamientos con MOID por debajo de un umbral.

    0.05 AU ≈ distancia estándar para clasificar un NEO como
    Potentially Hazardous Asteroid (PHA) por la NASA.

    Args:
        moid_series   : salida de compute_moid_timeseries()
        threshold_au  : umbral de distancia en AU (default: 0.05 AU)

    Returns:
        Lista de acercamientos, ordenados por MOID ascendente.
    """
    approaches = [m for m in moid_series if m["moid_au"] <= threshold_au]
    return sorted(approaches, key=lambda x: x["moid_au"])


# ── CONO DE INCERTIDUMBRE ─────────────────────────────────────────────────

def generate_uncertainty_cone(
    y0: np.ndarray,
    order: list[str],
    gm_map: dict,
    epoch_jd: float,
    dadt_mean: float,
    dadt_std: float,
    a_au: float,
    ecc: float,
    t_years: float = 40.0,
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    """
    Genera el cono de incertidumbre propagando N trayectorias con
    valores de da/dt muestreados de N(dadt_mean, dadt_std).

    Esto cuantifica cómo la incertidumbre en el parámetro de Yarkovsky
    se traduce en incertidumbre de posición a largo plazo.

    Args:
        y0         : vector de estado inicial (de pack_state_vector)
        order      : orden de cuerpos
        gm_map     : dict de GM
        epoch_jd   : época en JD
        dadt_mean  : da/dt central estimado [AU/My]
        dadt_std   : desviación estándar de da/dt [AU/My]
        a_au       : semieje mayor [AU]
        ecc        : excentricidad
        t_years    : horizonte de propagación [años]
        n_samples  : número de trayectorias del cono
        seed       : semilla aleatoria para reproducibilidad

    Returns:
        dict con claves:
            'trajectories'    : np.ndarray (n_samples, N_steps, 3)
            'dadt_samples'    : np.ndarray (n_samples,)
            'times_jd'        : np.ndarray (N_steps,)
            'pos_mean'        : np.ndarray (N_steps, 3)  — trayectoria media
            'pos_std'         : np.ndarray (N_steps, 3)  — desviación estándar
            'spread_au'       : np.ndarray (N_steps,)    — ancho del cono en AU
            'spread_km'       : np.ndarray (N_steps,)    — ancho del cono en km
    """
    rng = np.random.default_rng(seed)
    dadt_samples = rng.normal(dadt_mean, dadt_std, n_samples)

    print(f"[HYPATIA] Generando cono de incertidumbre: {n_samples} trayectorias, "
          f"{t_years:.0f} años, da/dt={dadt_mean:.3f}±{dadt_std:.3f} AU/My")

    trajectories = []
    times_jd_ref = None

    for i, dadt in enumerate(dadt_samples):
        A2 = dadt_to_A2(dadt, a_au, ecc)
        res = propagate_from_state(y0, order, gm_map, t_years, A2, epoch_jd)

        if times_jd_ref is None:
            times_jd_ref = res["times_jd"]

        # Interpolar al mismo grid de tiempo que la primera trayectoria
        # (pueden tener distinto n_steps si el paso adaptativo difiere)
        traj = res["asteroid_pos"]   # (N, 3)
        if len(traj) != len(times_jd_ref):
            # Re-muestrear al grid de referencia
            t_orig = res["times_days"]
            t_ref  = times_jd_ref - times_jd_ref[0]
            traj = np.column_stack([
                np.interp(t_ref, t_orig, traj[:, k]) for k in range(3)
            ])

        trajectories.append(traj)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples} trayectorias completadas")

    trajectories = np.array(trajectories)   # (n_samples, N_steps, 3)

    pos_mean = trajectories.mean(axis=0)    # (N_steps, 3)
    pos_std  = trajectories.std(axis=0)     # (N_steps, 3)
    spread_au = np.linalg.norm(pos_std, axis=1)   # (N_steps,)

    KM_PER_AU = 1.495978707e8
    spread_km = spread_au * KM_PER_AU

    print(f"[HYPATIA] Cono generado. "
          f"Ancho final a {t_years:.0f} años: "
          f"{spread_km[-1]:.0f} km ({spread_au[-1]*1e4:.2f} ×10⁻⁴ AU)")

    return {
        "trajectories": trajectories,
        "dadt_samples": dadt_samples,
        "times_jd"    : times_jd_ref,
        "pos_mean"    : pos_mean,
        "pos_std"     : pos_std,
        "spread_au"   : spread_au,
        "spread_km"   : spread_km,
    }


def cone_width_at_year(cone: dict, year_offset: float) -> float:
    """
    Retorna el ancho del cono (en km) en un año específico desde
    el inicio de la integración.

    Args:
        cone        : salida de generate_uncertainty_cone()
        year_offset : años desde el inicio (ej. 40.0)

    Returns:
        Ancho del cono en km en ese instante.
    """
    t_days = cone["times_jd"] - cone["times_jd"][0]
    target = year_offset * 365.25
    idx = np.argmin(np.abs(t_days - target))
    return float(cone["spread_km"][idx])
