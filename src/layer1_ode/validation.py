"""
validation.py
-------------
Valida el integrador de HYPATIA comparando la trayectoria propagada
contra las efemérides históricas de JPL Horizons.

Criterio de aceptación: RMSE de posición < 1000 km en un arco de 10 años.

Uso típico:
    from src.layer1_ode.validation import run_validation
    report = run_validation(asteroid_id=99942, epoch='2014-01-01')
    print(report['passed'], report['rmse_km'])
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


def fetch_ephemeris_arc(
    asteroid_id: int,
    epoch_start: str,
    epoch_end: str,
    step: str = "30d",
) -> dict:
    """
    Descarga las efemérides históricas de JPL Horizons para un arco
    de observación completo.

    Args:
        asteroid_id : ID JPL del asteroide
        epoch_start : inicio del arco 'YYYY-MM-DD'
        epoch_end   : fin del arco   'YYYY-MM-DD'
        step        : paso temporal (default: '30d')

    Returns:
        dict con:
            'times_jd' : np.ndarray, tiempos en JD
            'pos_au'   : np.ndarray (N, 3), posiciones en AU
    """
    obj = Horizons(
        id=str(asteroid_id),
        location="500@10",
        epochs={"start": epoch_start, "stop": epoch_end, "step": step},
    )
    vec = obj.vectors(refplane="ecliptic")

    times_jd = np.array(vec["datetime_jd"].data, dtype=float)
    pos_au = np.column_stack([
        np.array(vec["x"].data, dtype=float),
        np.array(vec["y"].data, dtype=float),
        np.array(vec["z"].data, dtype=float),
    ])

    print(f"[HYPATIA] Efemérides descargadas: {len(times_jd)} puntos "
          f"({epoch_start} → {epoch_end})")
    return {"times_jd": times_jd, "pos_au": pos_au}


def compute_position_errors(
    times_jd_pred: np.ndarray,
    pos_pred_au: np.ndarray,
    times_jd_ref: np.ndarray,
    pos_ref_au: np.ndarray,
) -> np.ndarray:
    """
    Calcula el error de posición en km entre la trayectoria predicha
    y la efeméride de referencia, interpolando al mismo grid temporal.

    Args:
        times_jd_pred : tiempos de la predicción en JD
        pos_pred_au   : posiciones predichas (N, 3) en AU
        times_jd_ref  : tiempos de la referencia en JD
        pos_ref_au    : posiciones de referencia (M, 3) en AU

    Returns:
        np.ndarray (M,): error de posición en km en cada punto de referencia
    """
    errors_km = []
    for i, t in enumerate(times_jd_ref):
        # Interpolar predicción al tiempo de referencia
        pos_interp = np.array([
            np.interp(t, times_jd_pred, pos_pred_au[:, k])
            for k in range(3)
        ])
        err_au = np.linalg.norm(pos_interp - pos_ref_au[i])
        errors_km.append(err_au * KM_PER_AU)

    return np.array(errors_km)


def run_validation(
    asteroid_id: int | str = 99942,
    epoch_start: str = "2014-01-01",
    arc_years: float = 10.0,
    dadt_au_my: float = 0.0,
    a_au: float = 0.9226,
    ecc: float = 0.1914,
    perturbers: list[str] = DEFAULT_PERTURBERS,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta la validación completa del integrador sobre un arco histórico.

    Flujo:
        1. Descargar efemérides históricas (ground truth)
        2. Propagar desde el inicio del arco con el integrador HYPATIA
        3. Comparar posiciones y calcular RMSE

    Args:
        asteroid_id  : ID JPL del asteroide (default: Apophis = 99942)
        epoch_start  : inicio del arco de validación
        arc_years    : duración del arco en años (default: 10)
        dadt_au_my   : da/dt a usar en la validación (0.0 = sin Yarkovsky)
        a_au         : semieje mayor del asteroide [AU]
        ecc          : excentricidad del asteroide
        perturbers   : cuerpos perturbadores
        verbose      : imprimir resumen del resultado

    Returns:
        dict con:
            'passed'       : bool, True si RMSE < 1000 km
            'rmse_km'      : float, RMSE de posición en km
            'mae_km'       : float, MAE de posición en km
            'max_error_km' : float, error máximo en km
            'errors_km'    : np.ndarray, errores punto a punto
            'n_points'     : int, número de puntos comparados
            'threshold_km' : float, umbral de aceptación
    """
    from astropy.time import Time

    epoch_end_dt = Time(epoch_start) + arc_years * 365.25
    epoch_end = epoch_end_dt.iso[:10]

    # Paso 1: Efemérides históricas (ground truth)
    ephemeris = fetch_ephemeris_arc(
        int(asteroid_id), epoch_start, epoch_end, step="30d"
    )

    # Paso 2: Condiciones iniciales y propagación
    ic = get_initial_conditions(asteroid_id, epoch_start, perturbers)
    y0, order, gm_map = pack_state_vector(ic)

    A2 = dadt_to_A2(dadt_au_my, a_au, ecc) if dadt_au_my != 0.0 else 0.0
    result = propagate_from_state(
        y0, order, gm_map,
        t_years=arc_years,
        A2=A2,
        epoch_jd=ic["epoch_jd"],
    )

    # Paso 3: Calcular errores
    errors_km = compute_position_errors(
        result["times_jd"], result["asteroid_pos"],
        ephemeris["times_jd"], ephemeris["pos_au"],
    )

    rmse_km = float(np.sqrt(np.mean(errors_km ** 2)))
    mae_km  = float(np.mean(errors_km))
    max_km  = float(np.max(errors_km))
    passed  = rmse_km < RMSE_THRESHOLD_KM

    if verbose:
        status = "✓ APROBADO" if passed else "✗ REVISAR"
        print(f"\n{'='*50}")
        print(f"  VALIDACIÓN DEL INTEGRADOR HYPATIA — {status}")
        print(f"{'='*50}")
        print(f"  Asteroide  : {asteroid_id}")
        print(f"  Arco       : {epoch_start} → {epoch_end} ({arc_years:.0f} años)")
        print(f"  da/dt      : {dadt_au_my:.4f} AU/My  (A2={A2:.3e} AU/día²)")
        print(f"  Puntos     : {len(errors_km)}")
        print(f"  RMSE       : {rmse_km:>10.1f} km  (umbral: {RMSE_THRESHOLD_KM:.0f} km)")
        print(f"  MAE        : {mae_km:>10.1f} km")
        print(f"  Error máx  : {max_km:>10.1f} km")
        print(f"{'='*50}\n")

    return {
        "passed"      : passed,
        "rmse_km"     : rmse_km,
        "mae_km"      : mae_km,
        "max_error_km": max_km,
        "errors_km"   : errors_km,
        "n_points"    : len(errors_km),
        "threshold_km": RMSE_THRESHOLD_KM,
    }


def compare_scenarios(
    y0: np.ndarray,
    order: list[str],
    gm_map: dict,
    epoch_jd: float,
    ephemeris: dict,
    dadt_values: dict[str, float],
    a_au: float,
    ecc: float,
    t_years: float = 40.0,
) -> dict:
    """
    Compara múltiples escenarios de da/dt contra la efeméride de referencia.

    Implementa el experimento central de HYPATIA:
        Escenario A: da/dt = 0.0  (sin Yarkovsky)
        Escenario B: da/dt = estimado por ML + series de tiempo
        Escenario C: da/dt = valor real medido (ground truth JPL)

    Args:
        y0           : vector de estado inicial
        order        : orden de cuerpos
        gm_map       : dict de GM
        epoch_jd     : época inicial en JD
        ephemeris    : dict con 'times_jd' y 'pos_au' del ground truth
        dadt_values  : dict {nombre_escenario: da/dt en AU/My}
                       ej. {'sin_yark': 0.0, 'hypatia': -0.19, 'jpl': -0.20}
        a_au         : semieje mayor [AU]
        ecc          : excentricidad
        t_years      : horizonte de integración [años]

    Returns:
        dict {nombre_escenario: {'rmse_km', 'mae_km', 'errors_km', 'pos'}}
    """
    results = {}

    for name, dadt in dadt_values.items():
        print(f"[HYPATIA] Escenario '{name}': da/dt={dadt:.4f} AU/My")
        A2 = dadt_to_A2(dadt, a_au, ecc) if dadt != 0.0 else 0.0
        res = propagate_from_state(y0, order, gm_map, t_years, A2, epoch_jd)

        errors_km = compute_position_errors(
            res["times_jd"], res["asteroid_pos"],
            ephemeris["times_jd"], ephemeris["pos_au"],
        )

        results[name] = {
            "rmse_km"  : float(np.sqrt(np.mean(errors_km ** 2))),
            "mae_km"   : float(np.mean(errors_km)),
            "errors_km": errors_km,
            "pos"      : res["asteroid_pos"],
            "times_jd" : res["times_jd"],
            "dadt"     : dadt,
        }

    # Calcular reducción porcentual respecto al escenario sin Yarkovsky
    if "sin_yark" in results:
        base_rmse = results["sin_yark"]["rmse_km"]
        for name, r in results.items():
            if name != "sin_yark" and base_rmse > 0:
                r["reduction_pct"] = (1 - r["rmse_km"] / base_rmse) * 100
            else:
                r["reduction_pct"] = 0.0

    print("\n[HYPATIA] Comparación de escenarios:")
    print(f"  {'Escenario':<15} {'RMSE (km)':>12} {'Reducción':>10}")
    print(f"  {'-'*40}")
    for name, r in results.items():
        red = f"{r.get('reduction_pct', 0):.1f}%" if name != "sin_yark" else "—"
        print(f"  {name:<15} {r['rmse_km']:>12.0f} {red:>10}")

    return results
