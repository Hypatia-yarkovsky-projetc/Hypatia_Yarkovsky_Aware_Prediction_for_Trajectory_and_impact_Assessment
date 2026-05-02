"""
initial_conditions.py
---------------------
Descarga condiciones iniciales de posición y velocidad desde la API
JPL Horizons para el asteroide objetivo y los planetas perturbadores.

Salida: vectores cartesianos en el sistema de referencia eclíptico J2000,
en unidades AU (posición) y AU/día (velocidad), centrado en el Sol.
"""

import numpy as np
from datetime import datetime
from astroquery.jplhorizons import Horizons
from .constants import JPL_IDS, DEFAULT_PERTURBERS


def _fetch_state_vector(body_id: int, epoch_jd: float) -> np.ndarray:
    """
    Descarga el vector de estado [x, y, z, vx, vy, vz] de un cuerpo
    en una época dada (fecha juliana) desde JPL Horizons.

    Args:
        body_id  : ID del cuerpo en JPL (ej. 99942 para Apophis)
        epoch_jd : Época en días julianos (TDB)

    Returns:
        np.ndarray de shape (6,): [x, y, z, vx, vy, vz]
        Unidades: AU y AU/día, centro: Sol (ID=10), marco: ECLJ2000
    """
    obj = Horizons(
        id=str(body_id),
        location="500@10",          # centro: Sol
        epochs=epoch_jd,
    )
    vec = obj.vectors(refplane="ecliptic")

    state = np.array([
        float(vec["x"][0]),
        float(vec["y"][0]),
        float(vec["z"][0]),
        float(vec["vx"][0]),
        float(vec["vy"][0]),
        float(vec["vz"][0]),
    ])
    return state


def get_initial_conditions(
    asteroid_id: int | str,
    epoch: str | float,
    perturbers: list[str] = DEFAULT_PERTURBERS,
) -> dict:
    """
    Obtiene condiciones iniciales para el asteroide y todos los
    cuerpos perturbadores en la época indicada.

    Args:
        asteroid_id : ID JPL del asteroide (ej. 99942 o '99942')
        epoch       : Fecha de inicio. Puede ser:
                      - str 'YYYY-MM-DD' (ej. '2024-01-01')
                      - float en días julianos (ej. 2460310.5)
        perturbers  : Lista de cuerpos perturbadores a incluir.
                      Por defecto: Sol + 5 planetas principales.

    Returns:
        dict con claves:
            'epoch_jd'    : float, época en días julianos
            'asteroid'    : np.ndarray (6,), estado del asteroide
            'perturbers'  : dict {nombre: np.ndarray (6,)}
            'gm_perturbers': dict {nombre: float}, GM en AU³/día²
    """
    from astropy.time import Time
    from .constants import GM

    # Convertir época a días julianos
    if isinstance(epoch, str):
        epoch_jd = float(Time(epoch, format="iso", scale="tdb").jd)
    else:
        epoch_jd = float(epoch)

    print(f"[HYPATIA] Descargando condiciones iniciales — época JD {epoch_jd:.2f}")

    # Estado del asteroide
    print(f"  → Asteroide ID={asteroid_id} ...")
    asteroid_state = _fetch_state_vector(int(asteroid_id), epoch_jd)

    # Estados de los cuerpos perturbadores
    perturber_states = {}
    for name in perturbers:
        if name not in JPL_IDS:
            raise ValueError(f"Cuerpo '{name}' no reconocido. "
                             f"Opciones: {list(JPL_IDS.keys())}")
        print(f"  → {name.capitalize()} ...")
        perturber_states[name] = _fetch_state_vector(JPL_IDS[name], epoch_jd)

    gm_perturbers = {name: GM[name] for name in perturbers}

    print(f"[HYPATIA] Condiciones iniciales descargadas correctamente.")
    return {
        "epoch_jd"      : epoch_jd,
        "asteroid"      : asteroid_state,
        "perturbers"    : perturber_states,
        "gm_perturbers" : gm_perturbers,
    }


def pack_state_vector(ic: dict) -> tuple[np.ndarray, list[str], dict]:
    """
    Empaqueta el diccionario de condiciones iniciales en un vector
    plano y = [ast_state | pert_1_state | pert_2_state | ...]
    listo para solve_ivp.

    Returns:
        y0     : np.ndarray (6 * (1 + n_perturbers),)
        order  : lista de nombres en el mismo orden que y0
                 (primer elemento siempre 'asteroid')
        gm_map : dict {nombre: GM}
    """
    order = ["asteroid"] + list(ic["perturbers"].keys())
    states = [ic["asteroid"]] + [ic["perturbers"][k] for k in order[1:]]
    y0 = np.concatenate(states)

    gm_map = {"asteroid": 0.0}          # el asteroide no atrae a los planetas
    gm_map.update(ic["gm_perturbers"])

    return y0, order, gm_map


def unpack_state_vector(y: np.ndarray, order: list[str]) -> dict:
    """
    Desempaqueta un vector de estado plano en un diccionario
    {nombre: np.ndarray (6,)}.
    """
    n = len(order)
    assert len(y) == 6 * n, f"Vector mal formado: {len(y)} != 6×{n}"
    return {order[i]: y[6*i : 6*i+6] for i in range(n)}
