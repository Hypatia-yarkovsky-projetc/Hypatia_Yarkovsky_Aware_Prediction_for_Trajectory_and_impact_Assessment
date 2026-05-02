"""
yarkovsky.py
------------
Implementación del efecto Yarkovsky como aceleración perturbativa.

Modelo: parametrización transversal (Vokrouhlický et al. 1999)
La fuerza actúa en dirección tangencial a la órbita con magnitud
que decae con el cuadrado de la distancia heliocéntrica.

Ecuación:
    a_Y = A2 * (r0 / r)² * t̂

donde:
    A2  : parámetro de Yarkovsky en AU/día²
    r0  : distancia de referencia = 1 AU
    r   : distancia heliocéntrica instantánea (AU)
    t̂   : vector unitario tangencial = v / |v|

Relación con da/dt (AU/My):
    da/dt ≈ 2·A2 / (n · a · √(1 − e²))

    n   : movimiento medio (rad/día) = √(GM_sol / a³)
    a   : semieje mayor (AU)
    e   : excentricidad
"""

import numpy as np
from .constants import R0_YARKOVSKY_AU, AU_MY_TO_AU_DAY, GM_SOL_AU3_DAY2


def yarkovsky_acceleration(
    pos: np.ndarray,
    vel: np.ndarray,
    A2: float,
    r0: float = R0_YARKOVSKY_AU,
) -> np.ndarray:
    """
    Calcula la aceleración de Yarkovsky en AU/día².

    Args:
        pos : posición heliocéntrica del asteroide [AU], shape (3,)
        vel : velocidad del asteroide [AU/día], shape (3,)
        A2  : parámetro de Yarkovsky [AU/día²]
              Positivo → el asteroide se aleja del Sol (da/dt > 0)
              Negativo → el asteroide se acerca al Sol (da/dt < 0)
        r0  : distancia de referencia [AU], por defecto 1 AU

    Returns:
        np.ndarray (3,): aceleración de Yarkovsky en AU/día²
        Retorna vector nulo si A2=0 o si la velocidad es despreciable.
    """
    if A2 == 0.0:
        return np.zeros(3)

    r_norm = np.linalg.norm(pos)
    v_norm = np.linalg.norm(vel)

    # Protección numérica: evitar división por cero
    if r_norm < 1e-10 or v_norm < 1e-15:
        return np.zeros(3)

    t_hat = vel / v_norm                        # vector tangencial unitario
    scale = A2 * (r0 / r_norm) ** 2            # magnitud de la perturbación

    return scale * t_hat


def dadt_to_A2(
    dadt_au_my: float,
    a_au: float,
    ecc: float,
) -> float:
    """
    Convierte la tasa de deriva orbital da/dt (AU/My) al parámetro
    A2 (AU/día²) usado en la ecuación de Yarkovsky.

    Args:
        dadt_au_my : da/dt en AU por millón de años
        a_au       : semieje mayor en AU
        ecc        : excentricidad orbital

    Returns:
        A2 en AU/día²

    Notas:
        da/dt ≈ 2·A2 / (n · a · √(1 − e²))
        ⟹  A2 = da/dt · n · a · √(1 − e²) / 2
    """
    dadt_au_day = dadt_au_my * AU_MY_TO_AU_DAY   # convertir a AU/día

    n = np.sqrt(GM_SOL_AU3_DAY2 / a_au ** 3)    # movimiento medio rad/día
    factor = np.sqrt(max(1 - ecc ** 2, 1e-10))  # √(1−e²)

    A2 = dadt_au_day * n * a_au * factor / 2.0
    return A2


def A2_to_dadt(
    A2: float,
    a_au: float,
    ecc: float,
) -> float:
    """
    Convierte el parámetro A2 (AU/día²) a da/dt (AU/My).

    Args:
        A2    : parámetro de Yarkovsky en AU/día²
        a_au  : semieje mayor en AU
        ecc   : excentricidad

    Returns:
        da/dt en AU/My
    """
    n = np.sqrt(GM_SOL_AU3_DAY2 / a_au ** 3)
    factor = np.sqrt(max(1 - ecc ** 2, 1e-10))

    dadt_au_day = 2.0 * A2 / (n * a_au * factor)
    return dadt_au_day / AU_MY_TO_AU_DAY         # convertir a AU/My


def yarkovsky_order_of_magnitude(diameter_km: float, a_au: float) -> float:
    """
    Estimación del orden de magnitud de |da/dt| para un asteroide dado.
    Útil para verificar que el valor estimado por ML es físicamente plausible.

    Fórmula simplificada de Vokrouhlický (1998):
        |da/dt| ~ C / (ρ · D · a^0.5)
    donde C ≈ 1.5e-4 AU·km·g/cm³ / My  (constante empírica)

    Args:
        diameter_km : diámetro del asteroide en km
        a_au        : semieje mayor en AU

    Returns:
        |da/dt| estimado en AU/My
    """
    rho_g_cm3 = 2.0        # densidad típica NEO tipo S [g/cm³]
    C = 0.14              # constante calibrada [AU·km·g/cm3/My]
    return C / (rho_g_cm3 * diameter_km * np.sqrt(a_au))
