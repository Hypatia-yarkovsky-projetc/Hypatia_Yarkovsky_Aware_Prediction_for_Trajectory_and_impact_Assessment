"""
residuals.py
------------
Pipeline central de la Capa 2 de HYPATIA.

Calcula la serie de residuos orbitales: diferencia entre el semieje
mayor observado (datos históricos JPL) y el predicho por el integrador
N-cuerpos sin Yarkovsky (Capa 1).

    ε(t) = a_observado(t) − a_predicho_sin_Yarkovsky(t)

Si el modelo fuera perfectamente gravitacional, ε(t) sería ruido blanco
puro con media cero. El efecto Yarkovsky introduce una tendencia
sistemática que crece aproximadamente de forma lineal con el tiempo.
Esa tendencia es la huella de da/dt que los métodos de regresión detectan.

Integración con Capa 1:
    - get_initial_conditions()   → condiciones iniciales del integrador
    - pack_state_vector()        → empaquetar estado para solve_ivp
    - propagate_from_state()     → propagar sin Yarkovsky (A2=0)
    - fetch_ephemeris_arc()      → descarga efemérides históricas
    - semi_major_axis()          → convertir [pos, vel] → semieje mayor
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# Importaciones de la Capa 1
from ..layer1_ode.initial_conditions import (
    get_initial_conditions,
    pack_state_vector,
)
from ..layer1_ode.integrator import propagate_from_state
from ..layer1_ode.validation import fetch_ephemeris_arc
from ..layer1_ode.utils import semi_major_axis, jd_to_iso, iso_to_jd
from ..layer1_ode.constants import DEFAULT_PERTURBERS, KM_TO_AU


# ── Dataclass de resultado ────────────────────────────────────────────────

@dataclass
class ResidualSeries:
    """
    Contenedor inmutable de la serie de residuos orbitales.

    Atributos:
        times_jd    : tiempos en días julianos (N,)
        times_years : tiempos en años desde el inicio de la serie (N,)
        a_obs       : semieje mayor observado en AU (N,)
        a_pred      : semieje mayor predicho sin Yarkovsky en AU (N,)
        epsilon     : residuos ε = a_obs − a_pred en AU (N,)
        epsilon_km  : residuos en km (N,)
        n_points    : número de puntos en la serie
        epoch_start : fecha de inicio 'YYYY-MM-DD'
        epoch_end   : fecha de fin   'YYYY-MM-DD'
        asteroid_id : ID JPL del asteroide
    """
    times_jd    : np.ndarray
    times_years : np.ndarray
    a_obs       : np.ndarray
    a_pred      : np.ndarray
    epsilon     : np.ndarray
    epsilon_km  : np.ndarray
    n_points    : int
    epoch_start : str
    epoch_end   : str
    asteroid_id : int

    def to_dataframe(self) -> pd.DataFrame:
        """Exporta la serie como DataFrame de pandas."""
        return pd.DataFrame({
            "times_jd"   : self.times_jd,
            "times_years": self.times_years,
            "a_obs_au"   : self.a_obs,
            "a_pred_au"  : self.a_pred,
            "epsilon_au" : self.epsilon,
            "epsilon_km" : self.epsilon_km,
        })

    def summary(self) -> str:
        lines = [
            f"ResidualSeries — Asteroide {self.asteroid_id}",
            f"  Arco       : {self.epoch_start} → {self.epoch_end}",
            f"  Puntos     : {self.n_points}",
            f"  ε media    : {self.epsilon.mean()*1e6:.4f} ×10⁻⁶ AU",
            f"  ε std      : {self.epsilon.std()*1e6:.4f} ×10⁻⁶ AU",
            f"  ε rango    : [{self.epsilon.min()*1e6:.4f}, "
                            f"{self.epsilon.max()*1e6:.4f}] ×10⁻⁶ AU",
        ]
        return "\n".join(lines)


# ── Función principal ─────────────────────────────────────────────────────

def build_residual_series(
    asteroid_id      : int | str,
    epoch_start      : str,
    epoch_end        : str,
    a_au             : float,
    ecc              : float,
    perturbers       : list[str] = DEFAULT_PERTURBERS,
    obs_step         : str = "30d",
    n_obs_limit      : Optional[int] = None,
    verbose          : bool = True,
) -> ResidualSeries:
    """
    Construye la serie de residuos orbitales comparando efemérides
    históricas de JPL con la predicción del integrador sin Yarkovsky.

    Flujo interno:
        1. Descarga efemérides históricas de JPL (posiciones observadas)
        2. Obtiene condiciones iniciales en epoch_start
        3. Propaga con A2=0 (sin Yarkovsky) → predicción pura gravitacional
        4. Convierte [pos, vel] → semieje mayor oscular en cada instante
        5. Calcula ε(t) = a_obs(t) − a_pred(t)

    Args:
        asteroid_id   : ID JPL del asteroide (ej. 99942 para Apophis)
        epoch_start   : inicio del arco 'YYYY-MM-DD'
        epoch_end     : fin del arco   'YYYY-MM-DD'
        a_au          : semieje mayor nominal del asteroide [AU]
        ecc           : excentricidad nominal
        perturbers    : cuerpos perturbadores a incluir
        obs_step      : paso entre observaciones ('30d', '60d', etc.)
        n_obs_limit   : si se especifica, simula tener solo las primeras
                        n_obs_limit observaciones (experimento de arco corto)
        verbose       : imprimir progreso

    Returns:
        ResidualSeries con todos los arrays calculados
    """
    if verbose:
        print(f"\n[HYPATIA L2] Construyendo serie de residuos")
        print(f"  Asteroide  : {asteroid_id}")
        print(f"  Arco       : {epoch_start} → {epoch_end}")
        if n_obs_limit:
            print(f"  Simulando  : solo {n_obs_limit} observaciones iniciales")

    # ── Paso 1: Efemérides observadas (ground truth) ──────────────────────
    ephemeris = fetch_ephemeris_arc(
        int(asteroid_id), epoch_start, epoch_end, step=obs_step
    )
    times_jd_obs = ephemeris["times_jd"]
    pos_obs      = ephemeris["pos_au"]      # (N, 3) — solo posición

    # También necesitamos velocidades para calcular el semieje mayor
    # Las descargamos por separado
    from astroquery.jplhorizons import Horizons
    obj = Horizons(
        id=str(asteroid_id), location="500@10",
        epochs={"start": epoch_start, "stop": epoch_end, "step": obs_step},
    )
    vec = obj.vectors(refplane="ecliptic")
    vel_obs = np.column_stack([
        np.array(vec["vx"].data, dtype=float),
        np.array(vec["vy"].data, dtype=float),
        np.array(vec["vz"].data, dtype=float),
    ])

    # Aplicar límite de observaciones (experimento de arco corto)
    if n_obs_limit and n_obs_limit < len(times_jd_obs):
        times_jd_obs = times_jd_obs[:n_obs_limit]
        pos_obs      = pos_obs[:n_obs_limit]
        vel_obs      = vel_obs[:n_obs_limit]
        if verbose:
            print(f"  Recortando a {n_obs_limit} obs "
                  f"(hasta {jd_to_iso(times_jd_obs[-1])})")

    # ── Paso 2: Condiciones iniciales en epoch_start ──────────────────────
    ic = get_initial_conditions(asteroid_id, epoch_start, perturbers)
    y0, order, gm_map = pack_state_vector(ic)
    epoch_jd = ic["epoch_jd"]

    # ── Paso 3: Propagación sin Yarkovsky ────────────────────────────────
    t_years = (times_jd_obs[-1] - epoch_jd) / 365.25
    result = propagate_from_state(
        y0, order, gm_map,
        t_years=t_years,
        A2=0.0,
        epoch_jd=epoch_jd,
    )

    t_pred = result["times_jd"]       # (M,)
    pos_pred_arr = result["asteroid_pos"]   # (M, 3)
    vel_pred_arr = result["asteroid_vel"]   # (M, 3)

    # ── Paso 4: Semieje mayor oscular en cada punto ───────────────────────
    # Observado: desde las efemérides JPL
    a_obs_arr = np.array([
        semi_major_axis(pos_obs[i], vel_obs[i])
        for i in range(len(times_jd_obs))
    ])

    # Predicho: interpolar predicción al mismo grid temporal que obs
    a_pred_arr = np.zeros(len(times_jd_obs))
    for i, t in enumerate(times_jd_obs):
        # Interpolar posición y velocidad predichas al tiempo observado
        pos_i = np.array([np.interp(t, t_pred, pos_pred_arr[:, k]) for k in range(3)])
        vel_i = np.array([np.interp(t, t_pred, vel_pred_arr[:, k]) for k in range(3)])
        a_pred_arr[i] = semi_major_axis(pos_i, vel_i)

    # ── Paso 5: Residuos ε(t) = a_obs − a_pred ───────────────────────────
    epsilon = a_obs_arr - a_pred_arr
    epsilon_km = epsilon * 1.495978707e8   # AU → km

    # Tiempo en años desde el inicio
    times_years = (times_jd_obs - times_jd_obs[0]) / 365.25

    if verbose:
        n = len(times_jd_obs)
        print(f"\n[HYPATIA L2] Serie de residuos construida: {n} puntos")
        print(f"  Tendencia visible: {np.polyfit(times_years, epsilon, 1)[0]*1e6:.3f} ×10⁻⁶ AU/año")

    return ResidualSeries(
        times_jd    = times_jd_obs,
        times_years = times_years,
        a_obs       = a_obs_arr,
        a_pred      = a_pred_arr,
        epsilon     = epsilon,
        epsilon_km  = epsilon_km,
        n_points    = len(times_jd_obs),
        epoch_start = epoch_start,
        epoch_end   = jd_to_iso(times_jd_obs[-1]),
        asteroid_id = int(asteroid_id),
    )


def simulate_short_arc(
    full_series: ResidualSeries,
    n_obs: int,
) -> ResidualSeries:
    """
    Simula el escenario de objeto recién descubierto recortando
    la serie completa a las primeras n_obs observaciones.

    Útil para el experimento central: comparar precisión de estimación
    de da/dt con 5, 10, 20 observaciones vs el arco completo.

    Args:
        full_series : ResidualSeries completa
        n_obs       : número de observaciones a conservar

    Returns:
        Nueva ResidualSeries recortada
    """
    n = min(n_obs, full_series.n_points)
    return ResidualSeries(
        times_jd    = full_series.times_jd[:n],
        times_years = full_series.times_years[:n],
        a_obs       = full_series.a_obs[:n],
        a_pred      = full_series.a_pred[:n],
        epsilon     = full_series.epsilon[:n],
        epsilon_km  = full_series.epsilon_km[:n],
        n_points    = n,
        epoch_start = full_series.epoch_start,
        epoch_end   = jd_to_iso(full_series.times_jd[n-1]),
        asteroid_id = full_series.asteroid_id,
    )
