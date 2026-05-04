"""
initial_conditions.py
Descarga condiciones iniciales de posición y velocidad desde la API
JPL Horizons para el asteroide objetivo y los planetas perturbadores.
Salida: vectores cartesianos en el sistema de referencia eclíptico J2000,
en unidades AU (posición) y AU/día (velocidad), centrado en el Sol.
"""
import numpy as np
import concurrent.futures
from functools import lru_cache
from astroquery.jplhorizons import Horizons
from .constants import JPL_IDS, DEFAULT_PERTURBERS

@lru_cache(maxsize=128)
def _fetch_state_vector(body_id: int, epoch_jd: float) -> np.ndarray:
    """
    Descarga el vector de estado [x, y, z, vx, vy, vz] de un cuerpo
    en una época dada desde JPL Horizons con timeout de 30s.
    """
    def _get_table():
        obj = Horizons(
            id=str(body_id),
            location="@10",
            epochs=epoch_jd,
        )
        return obj.vectors(refplane="ecliptic")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_get_table)
        try:
            vec = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(f"Timeout conectando con JPL Horizons para cuerpo {body_id}")

    return np.array([
        float(vec["x"][0]),
        float(vec["y"][0]),
        float(vec["z"][0]),
        float(vec["vx"][0]),
        float(vec["vy"][0]),
        float(vec["vz"][0]),
    ], dtype=float)

def get_initial_conditions(
    asteroid_id: int | str,
    epoch: str | float,
    perturbers: list[str] = DEFAULT_PERTURBERS,
) -> dict:
    """
    Obtiene condiciones iniciales para el asteroide y todos los
    cuerpos perturbadores en la época indicada.
    """
    from astropy.time import Time
    from .constants import GM

    epoch_jd = float(Time(epoch, format="iso", scale="tdb").jd) if isinstance(epoch, str) else float(epoch)

    print(f"[HYPATIA] Descargando condiciones iniciales — época JD {epoch_jd:.2f}")
    print(f"  → Asteroide ID={asteroid_id} ...")
    asteroid_state = _fetch_state_vector(int(asteroid_id), epoch_jd)

    perturber_states = {}
    for name in perturbers:
        if name not in JPL_IDS:
            raise ValueError(f"Cuerpo '{name}' no reconocido.")
        print(f"  → {name.capitalize()} ...")
        perturber_states[name] = _fetch_state_vector(JPL_IDS[name], epoch_jd)

    gm_perturbers = {name: GM[name] for name in perturbers}
    print("[HYPATIA] Condiciones iniciales descargadas correctamente.")

    return {
        "epoch_jd": epoch_jd,
        "asteroid": asteroid_state,
        "perturbers": perturber_states,
        "gm_perturbers": gm_perturbers,
    }

def pack_state_vector(ic: dict) -> tuple[np.ndarray, list[str], dict]:
    """Empaqueta el diccionario de condiciones iniciales en un vector plano."""
    order = ["asteroid"] + list(ic["perturbers"].keys())
    states = [ic["asteroid"]] + [ic["perturbers"][k] for k in order[1:]]
    y0 = np.concatenate(states)

    gm_map = {"asteroid": 0.0}
    gm_map.update(ic["gm_perturbers"])

    return y0, order, gm_map

def unpack_state_vector(y: np.ndarray, order: list[str]) -> dict:
    """Desempaqueta un vector de estado plano en un diccionario."""
    n = len(order)
    assert len(y) == 6 * n, f"Vector mal formado: {len(y)} != 6×{n}"
    return {order[i]: y[6*i : 6*i+6] for i in range(n)}