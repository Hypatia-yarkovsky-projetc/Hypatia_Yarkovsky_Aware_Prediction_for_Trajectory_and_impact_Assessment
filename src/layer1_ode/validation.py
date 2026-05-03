"""
validation.py
Validación contra JPL Horizons + comparación de escenarios.
Incluye controles estrictos de unidades, alineación temporal e interpolación segura.
"""
import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from .initial_conditions import get_initial_conditions, pack_state_vector
from .integrator import propagate_from_state
from .yarkovsky import dadt_to_A2
from .constants import DEFAULT_PERTURBERS

KM_PER_AU = 1.495978707e8

def fetch_ephemeris_arc(asteroid_id, epoch_start, epoch_end, step="30d"):
    """Descarga efemérides de JPL Horizons en unidades AU explícitas."""
    obj = Horizons(id=str(asteroid_id), location="@10", epochs={"start": epoch_start, "stop": epoch_end, "step": step})
    vec = obj.vectors(refplane="ecliptic")

    # Extracción segura de columnas
    times = np.asarray(vec["datetime_jd"].data, dtype=float)
    x = np.asarray(vec["x"].data, dtype=float)
    y = np.asarray(vec["y"].data, dtype=float)
    z = np.asarray(vec["z"].data, dtype=float)

    # Detección y corrección automática de unidades
    # Horizons suele devolver AU, pero versiones recientes o configuraciones pueden devolver km.
    mean_dist = np.mean(np.sqrt(x**2 + y**2 + z**2))
    if mean_dist > 10.0:  # Claramente en km
        x /= KM_PER_AU
        y /= KM_PER_AU
        z /= KM_PER_AU

    pos = np.column_stack([x, y, z])

    # Diagnóstico rápido
    print(f"[VALIDATION] JPL Ephemeris loaded: {len(times)} points | JD range: {times.min():.2f} - {times.max():.2f}")
    print(f"[VALIDATION] Position norm range: {np.linalg.norm(pos, axis=1).min():.4f} - {np.linalg.norm(pos, axis=1).max():.4f} AU")

    return {"times_jd": times, "pos_au": pos}

def compute_position_errors(times_pred, pos_pred, times_ref, pos_ref):
    """Calcula errores de posición con interpolación segura y validación de rangos."""
    times_pred = np.asarray(times_pred, dtype=float)
    pos_pred = np.asarray(pos_pred, dtype=float)
    times_ref = np.asarray(times_ref, dtype=float)
    pos_ref = np.asarray(pos_ref, dtype=float)

    if pos_pred.ndim != 2 or pos_pred.shape[1] != 3:
        raise ValueError(f"pos_pred debe tener forma (N, 3). Obtenido: {pos_pred.shape}")
    if pos_ref.ndim != 2 or pos_ref.shape[1] != 3:
        raise ValueError(f"pos_ref debe tener forma (M, 3). Obtenido: {pos_ref.shape}")

    # Diagnóstico de alineación temporal
    print(f"[VALIDATION] Pred time range: {times_pred.min():.2f} - {times_pred.max():.2f} JD")
    print(f"[VALIDATION] Ref  time range: {times_ref.min():.2f} - {times_ref.max():.2f} JD")

    t_min, t_max = times_pred.min(), times_pred.max()
    # Filtrar puntos de referencia estrictamente dentro del rango de propagación
    valid_mask = (times_ref >= t_min) & (times_ref <= t_max)
    if not np.any(valid_mask):
        raise ValueError("No hay solapamiento temporal entre la propagación y la efeméride de referencia. Verifica épocas y escalas de tiempo.")

    t_ref_valid = times_ref[valid_mask]
    p_ref_valid = pos_ref[valid_mask]

    errors = np.empty(len(t_ref_valid))
    for i, t in enumerate(t_ref_valid):
        # Interpolación segura (np.interp requiere xp ordenado, que times_pred ya lo está)
        pos_i = np.array([np.interp(t, times_pred, pos_pred[:, k]) for k in range(3)])
        errors[i] = np.linalg.norm(pos_i - p_ref_valid[i])

    # Convertir a km
    return errors * KM_PER_AU, valid_mask

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
    errors_km, _ = compute_position_errors(res["times_jd"], res["asteroid_pos"], eph["times_jd"], eph["pos_au"])

    rmse = float(np.sqrt(np.mean(errors_km**2)))
    report = {"passed": rmse < 1000.0, "rmse_km": rmse, "mae_km": float(np.mean(errors_km)),
              "max_error_km": float(np.max(errors_km)), "errors_km": errors_km, "n_points": len(errors_km)}

    if verbose:
        status = "PASA" if report["passed"] else "REVISAR"
        print(f"\n{'='*50}\n  VALIDACION HYPATIA — {status}\n{'='*50}")
        print(f"  RMSE: {rmse:8.1f} km | Umbral: 1000 km | Puntos: {len(errors_km)}\n{'='*50}")
    return report

def compare_scenarios(y0, order, gm_map, epoch_jd, ephemeris, dadt_values, a_au, ecc, t_years=40.0, drift_cutoff_years=10.0):
    """
    Compara escenarios de propagación orbital.
    Evalúa RMSE exclusivamente sobre el horizonte donde el drift secular es significativo.
    """
    cutoff_jd = epoch_jd + drift_cutoff_years * 365.25
    ref_times = np.asarray(ephemeris["times_jd"], dtype=float)
    ref_pos = np.asarray(ephemeris["pos_au"], dtype=float)

    # Filtrar horizonte futuro
    mask_future = ref_times >= cutoff_jd
    if not np.any(mask_future):
        raise ValueError(f"No hay puntos de referencia después de JD {cutoff_jd:.2f}. Verifica t_years y drift_cutoff_years.")

    ref_times_cut = ref_times[mask_future]
    ref_pos_cut = ref_pos[mask_future]

    results = {}
    for name, dadt in dadt_values.items():
        A2 = dadt_to_A2(dadt, a_au, ecc) if dadt != 0.0 else 0.0
        res = propagate_from_state(y0, order, gm_map, t_years, A2, epoch_jd)

        errors_km, valid_mask = compute_position_errors(
            res["times_jd"], res["asteroid_pos"],
            ref_times_cut, ref_pos_cut
        )

        if len(errors_km) == 0:
            rmse = mae = max_err = 0.0
        else:
            rmse = float(np.sqrt(np.mean(errors_km**2)))
            mae = float(np.mean(errors_km))
            max_err = float(np.max(errors_km))

        results[name] = {
            "rmse_km": rmse,
            "mae_km": mae,
            "max_error_km": max_err,
            "n_points_eval": len(errors_km),
            "pos": res["asteroid_pos"]
        }
        print(f"[VALIDATION] {name}: RMSE = {rmse:,.0f} km ({len(errors_km)} pts)")

    base_rmse = results.get("sin_yark", {}).get("rmse_km", 0.0)
    for name, r in results.items():
        if name != "sin_yark" and base_rmse > 0:
            r["reduction_pct"] = ((base_rmse - r["rmse_km"]) / base_rmse) * 100.0
        else:
            r["reduction_pct"] = 0.0

    return results