"""
moid.py
MOID y cono de incertidumbre orbital.
"""
import numpy as np
from tqdm import tqdm
from .integrator import propagate_from_state
from .yarkovsky import dadt_to_A2

def compute_moid_timeseries(result_ast: dict, result_earth: dict, window_years: float = 5.0) -> list[dict]:
    times = result_ast["times_jd"]
    pos_ast = result_ast["asteroid_pos"]
    pos_ear = result_earth.get("asteroid_pos") or result_earth.get("earth_pos")
    if pos_ear is None: raise ValueError("Resultado de Tierra sin clave 'asteroid_pos' o 'earth_pos'")
    
    dist_au = np.linalg.norm(pos_ast - pos_ear, axis=1)
    window_days = window_years * 365.25
    t_start, t_end = times[0], times[-1]
    results = []
    t = t_start
    while t < t_end:
        mask = (times >= t) & (times < t + window_days)
        if mask.sum() < 2:
            t += window_days
            continue
        idx_loc = np.argmin(dist_au[mask])
        idx_glob = np.where(mask)[0][idx_loc]
        results.append({
            "t_center_jd": times[mask][idx_loc], "moid_au": float(dist_au[idx_glob]),
            "moid_km": float(dist_au[idx_glob] * 1.495978707e8), "idx_min": int(idx_glob)
        })
        t += window_days
    return results

def find_close_approaches(moid_series: list[dict], threshold_au: float = 0.05) -> list[dict]:
    return sorted([m for m in moid_series if m["moid_au"] <= threshold_au], key=lambda x: x["moid_au"])

def generate_uncertainty_cone(y0, order, gm_map, epoch_jd, dadt_mean, dadt_std, a_au, ecc, t_years=40.0, n_samples=50, seed=42):
    rng = np.random.default_rng(seed)
    dadt_samples = rng.normal(dadt_mean, dadt_std, n_samples)
    trajectories, times_ref = [], None
    
    print(f"[HYPATIA] Generando cono: {n_samples} trayectorias, {t_years} años")
    for i, dadt in enumerate(tqdm(dadt_samples, desc="Propagando")):
        A2 = dadt_to_A2(dadt, a_au, ecc)
        res = propagate_from_state(y0, order, gm_map, t_years, A2, epoch_jd)
        if times_ref is None: times_ref = res["times_jd"]
        
        t_orig = res["times_jd"]
        if len(t_orig) != len(times_ref):
            traj = np.column_stack([np.interp(times_ref, t_orig, res["asteroid_pos"][:, k]) for k in range(3)])
        else:
            traj = res["asteroid_pos"]
        trajectories.append(traj)
        
    traj_arr = np.array(trajectories)
    spread_km = np.linalg.norm(traj_arr.std(axis=0), axis=1) * 1.495978707e8
    
    return {
        "trajectories": traj_arr, "dadt_samples": dadt_samples, "times_jd": times_ref,
        "pos_mean": traj_arr.mean(axis=0), "pos_std": traj_arr.std(axis=0),
        "spread_km": spread_km
    }

def cone_width_at_year(cone: dict, year_offset: float) -> float:
    target = year_offset * 365.25
    idx = np.argmin(np.abs(cone["times_jd"] - cone["times_jd"][0] - target))
    return float(cone["spread_km"][idx])