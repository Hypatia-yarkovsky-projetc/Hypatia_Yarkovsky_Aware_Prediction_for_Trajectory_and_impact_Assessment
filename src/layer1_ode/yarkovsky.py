"""
yarkovsky.py
Modelo de Yarkovsky parametrizado transversal (Vokrouhlický et al. 1999)
"""
import numpy as np
from .constants import R0_YARKOVSKY_AU, AU_MY_TO_AU_DAY, GM_SOL_AU3_DAY2

def yarkovsky_acceleration(pos: np.ndarray, vel: np.ndarray, A2: float, r0: float = R0_YARKOVSKY_AU) -> np.ndarray:
    if A2 == 0.0:
        return np.zeros(3)
    r_norm = np.linalg.norm(pos)
    v_norm = np.linalg.norm(vel)
    if r_norm < 1e-10 or v_norm < 1e-15:
        return np.zeros(3)
    return A2 * (r0 / r_norm)**2 * (vel / v_norm)

def dadt_to_A2(dadt_au_my: float, a_au: float, ecc: float) -> float:
    dadt_au_day = dadt_au_my * AU_MY_TO_AU_DAY
    n = np.sqrt(GM_SOL_AU3_DAY2 / a_au**3)
    factor = np.sqrt(max(1.0 - ecc**2, 1e-10))
    return dadt_au_day * n * a_au * factor / 2.0

def A2_to_dadt(A2: float, a_au: float, ecc: float) -> float:
    n = np.sqrt(GM_SOL_AU3_DAY2 / a_au**3)
    factor = np.sqrt(max(1.0 - ecc**2, 1e-10))
    return (2.0 * A2 / (n * a_au * factor)) / AU_MY_TO_AU_DAY

def yarkovsky_order_of_magnitude(diameter_km: float, a_au: float) -> float:
    """Estimación empírica calibrada para NEOs tipo S/C"""
    rho_g_cm3 = 2.0
    C = 0.14  # AU·km·g/cm³/My (ajustado a observaciones de Apophis/Bennu)
    return C / (rho_g_cm3 * diameter_km * np.sqrt(a_au))