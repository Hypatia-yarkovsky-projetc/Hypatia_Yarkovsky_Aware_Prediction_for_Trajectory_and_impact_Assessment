"""
validation.py
Validación contra JPL Horizons + comparación de escenarios.
"""
import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from .initial_conditions import get_initial_conditions, pack_state_vector
from .integrator import propagate_from_state
from .yarkovsky import dadt_to_A2
from .constants import DEFAULT_PERTURBERS

KM_PER_AU = 1.495978707e8
RMSE_THRESHOLD_KM = 1000.0

def fetch_ephemeris_arc(asteroid_id, epoch_start, epoch_end, step="30d"):
    obj = Horizons(id=str(asteroid_id), location="@10", epochs={"start": epoch_start, "stop": epoch_end, "step": step})
    vec = obj.vectors(refplane="ecliptic", out_type="NO")
    return {
        "times_jd": np.array(vec["datetime_jd"].data, dtype=float),
        "pos_au": np.column_stack([vec["x"].data, vec["y"].data, vec["z"].data])
    }

def compute_position_errors(times_pred, pos_pred, times_ref, pos_ref):
    errors = np.empty(len(times_ref))
    for i, t in enumerate(times_ref):
        pos_i = np.array([np.interp(t, times_pred, pos_pred[:, k]) for k in range(3)])
        errors[i] = np.linalg.norm(pos_i - pos_ref[i]) * KM_PER_AU
    return errors

def run_validation(asteroid_id=99942, epoch_start="2014-01-01", arc_years=10.0, 
                   dadt_au_my=0.0, a_au=0.9226, ecc=0.1914, perturbers=DEFAULT_PERTURBERS, verbose=True):
    epoch_jd_start = Time(epoch_start, scale="tdb").jd
    epoch_jd_end = epoch_jd_start + arc_years * 365.25
    epoch_end_iso = Time(epoch_jd_end, format="jd").iso[:10]

    eph = fetch_ephemeris_arc(asteroid_id, epoch_start, epoch_end_iso, step="30d")
    ic = get_initial_conditions(asteroid_id, epoch_start, perturbers)
    y0, order, gm_map = pack_state_vector(ic)
    A2 = dadt_to_A2(dadt_au_my, a_au, ecc) if dadt_au_my != 0.0 else 0.0
    
    res = propagate_from_state(y0, order, gm_map, arc_years, A2, ic["epoch_jd"])
    errors_km = compute_position_errors(res["times_jd"], res["asteroid_pos"], eph["times_jd"], eph["pos_au"])

    rmse = float(np.sqrt(np.mean(errors_km**2)))
    report = {"passed": rmse < RMSE_THRESHOLD_KM, "rmse_km": rmse, "mae_km": float(np.mean(errors_km)),
              "max_error_km": float(np.max(errors_km)), "errors_km": errors_km, "n_points": len(errors_km)}
    
    if verbose:
        print(f"\n{'='*50}\n  VALIDACIÓN HYPATIA — {'✓ PASA' if report['passed'] else '✗ REVISAR'}\n{'='*50}")
        print(f"  RMSE: {rmse:8.1f} km | Umbral: {RMSE_THRESHOLD_KM} km | Puntos: {len(errors_km)}\n{'='*50}")
    return report

def compare_scenarios(y0, order, gm_map, epoch_jd, ephemeris, dadt_values, a_au, ecc, t_years=40.0):
    results = {}
    for name, dadt in dadt_values.items():
        A2 = dadt_to_A2(dadt, a_au, ecc) if dadt != 0.0 else 0.0
        res = propagate_from_state(y0, order, gm_map, t_years, A2, epoch_jd)
        errors = compute_position_errors(res["times_jd"], res["asteroid_pos"], ephemeris["times_jd"], ephemeris["pos_au"])
        results[name] = {"rmse_km": float(np.sqrt(np.mean(errors**2))), "mae_km": float(np.mean(errors)), "pos": res["asteroid_pos"]}
    
    base = results.get("sin_yark", {}).get("rmse_km", 0)
    for name, r in results.items():
        r["reduction_pct"] = (1 - r["rmse_km"]/base)*100 if name != "sin_yark" and base > 0 else 0.0
    return results