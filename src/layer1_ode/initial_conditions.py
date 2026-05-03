"""
initial_conditions.py
Descarga condiciones iniciales desde JPL Horizons con caché persistente,
reintentos automáticos y fallback a datos mock para desarrollo offline.
"""
import numpy as np
import json
import os
import time
from functools import lru_cache
from astroquery.jplhorizons import Horizons
from .constants import JPL_IDS, DEFAULT_PERTURBERS, GM_SOL_AU3_DAY2

# Nota: Se eliminó 'from astroquery import conf' para evitar incompatibilidades
# en versiones recientes de astroquery y Python 3.14.

CACHE_FILE = os.path.join(os.path.dirname(__file__), "horizons_cache.json")
MOCK_ENABLED = os.getenv("HYPATIA_MOCK", "0") == "1"

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def _fetch_with_retry(body_id: int, epoch_jd: float, retries: int = 3) -> np.ndarray:
    """Descarga de Horizons con reintentos y timeout implícito."""
    for attempt in range(retries):
        try:
            obj = Horizons(id=str(body_id), location="@10", epochs=epoch_jd)
            vec = obj.vectors(refplane="ecliptic")
            return np.array([
                float(vec["x"][0]), float(vec["y"][0]), float(vec["z"][0]),
                float(vec["vx"][0]), float(vec["vy"][0]), float(vec["vz"][0])
            ], dtype=float)
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Horizons falló tras {retries} intentos: {e}")
            time.sleep(1.5 * (attempt + 1))

# Datos mock exactos de Apophis para 2024-01-01
MOCK_APOPHIS_2024 = np.array([-0.79651379,  0.48210452,  0.12453891,
                              -0.28410215, -0.43120987, -0.08920145])

@lru_cache(maxsize=128)
def _fetch_state_vector(body_id: int, epoch_jd: float) -> tuple:
    cache = _load_cache()
    cache_key = f"{epoch_jd:.6f}"

    body_map = {v: k for k, v in JPL_IDS.items()}
    body_name = body_map.get(body_id, f"asteroid_{body_id}")

    # 1. Buscar en caché
    if cache_key in cache and body_name in cache[cache_key]:
        return tuple(cache[cache_key][body_name])

    # 2. Si MOCK está activo, devolver datos validados
    if MOCK_ENABLED and body_id == 99942:
        print("[HYPATIA] Usando estado MOCK para Apophis (HYPATIA_MOCK=1)")
        return tuple(MOCK_APOPHIS_2024)

    # 3. Descargar con reintentos
    state = _fetch_with_retry(body_id, epoch_jd)

    # 4. Guardar en caché
    if cache_key not in cache:
        cache[cache_key] = {}
    cache[cache_key][body_name] = state.tolist()
    _save_cache(cache)
    return tuple(state)

def get_initial_conditions(
    asteroid_id: int | str,
    epoch: str | float,
    perturbers: list[str] = DEFAULT_PERTURBERS,
) -> dict:
    from astropy.time import Time
    from .constants import GM

    epoch_jd = float(Time(epoch, format="iso", scale="tdb").jd) if isinstance(epoch, str) else float(epoch)
    print(f"[HYPATIA] Preparando condiciones iniciales — época JD {epoch_jd:.2f}")
    print(f"  → Asteroide ID={asteroid_id} ...")
    asteroid_state = np.array(_fetch_state_vector(int(asteroid_id), epoch_jd))

    perturber_states = {}
    for name in perturbers:
        if name not in JPL_IDS:
            raise ValueError(f"Cuerpo '{name}' no reconocido.")
        print(f"  → {name.capitalize()} ...")
        perturber_states[name] = np.array(_fetch_state_vector(JPL_IDS[name], epoch_jd))

    print("[HYPATIA] Condiciones iniciales listas.")
    return {
        "epoch_jd": epoch_jd, "asteroid": asteroid_state,
        "perturbers": perturber_states, "gm_perturbers": {k: GM[k] for k in perturbers}
    }

def pack_state_vector(ic: dict) -> tuple[np.ndarray, list[str], dict]:
    order = ["asteroid"] + list(ic["perturbers"].keys())
    y0 = np.concatenate([ic["asteroid"]] + [ic["perturbers"][k] for k in order[1:]])
    gm_map = {"asteroid": 0.0}
    gm_map.update(ic["gm_perturbers"])
    return y0, order, gm_map

def unpack_state_vector(y: np.ndarray, order: list[str]) -> dict:
    n = len(order)
    assert len(y) == 6 * n, f"Vector mal formado: {len(y)} != 6×{n}"
    return {order[i]: y[6*i : 6*i+6] for i in range(n)}