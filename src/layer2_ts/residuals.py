"""
residuals.py
Pipeline central de la Capa 2.
Calcula la serie de residuos orbitales epsilon(t) = a_obs(t) - a_pred(t).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from astroquery.jplhorizons import Horizons

from ..layer1_ode.initial_conditions import get_initial_conditions, pack_state_vector
from ..layer1_ode.integrator import propagate_from_state
from ..layer1_ode.utils import semi_major_axis, jd_to_iso
from ..layer1_ode.constants import DEFAULT_PERTURBERS

@dataclass
class ResidualSeries:
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
        return pd.DataFrame({
            "times_jd": self.times_jd, "times_years": self.times_years,
            "a_obs_au": self.a_obs, "a_pred_au": self.a_pred,
            "epsilon_au": self.epsilon, "epsilon_km": self.epsilon_km,
        })

    def summary(self) -> str:
        return (
            f"ResidualSeries - Asteroide {self.asteroid_id}\n"
            f"  Arco   : {self.epoch_start} -> {self.epoch_end}\n"
            f"  Puntos : {self.n_points}\n"
            f"  Epsilon: media={self.epsilon.mean()*1e6:.4f}x10^-6 AU\n"
        )

def build_residual_series(
    asteroid_id : int | str,
    epoch_start : str,
    epoch_end   : str,
    a_au        : float,
    ecc         : float,
    perturbers  : list[str] = DEFAULT_PERTURBERS,
    obs_step    : str = "30d",
    n_obs_limit : Optional[int] = None,
    verbose     : bool = True,
) -> ResidualSeries:
    """Construye la serie de residuos comparando efemerides JPL con integrador sin Yarkovsky."""
    if verbose:
        print(f"[HYPATIA L2] Construyendo serie de residuos: {asteroid_id} | {epoch_start} -> {epoch_end}")

    # Solicitud unica a Horizons: posicion y velocidad
    obj = Horizons(id=str(asteroid_id), location="500@10",
                   epochs={"start": epoch_start, "stop": epoch_end, "step": obs_step})
    vec = obj.vectors(refplane="ecliptic")
    
    times_jd_obs = np.array(vec["datetime_jd"].data, dtype=float)
    pos_obs = np.column_stack([vec["x"].data, vec["y"].data, vec["z"].data]).astype(float)
    vel_obs = np.column_stack([vec["vx"].data, vec["vy"].data, vec["vz"].data]).astype(float)

    # Recorte de observaciones si se requiere
    if n_obs_limit and n_obs_limit < len(times_jd_obs):
        times_jd_obs = times_jd_obs[:n_obs_limit]
        pos_obs = pos_obs[:n_obs_limit]
        vel_obs = vel_obs[:n_obs_limit]

    # Condiciones iniciales y propagacion gravitacional pura
    ic = get_initial_conditions(asteroid_id, epoch_start, perturbers)
    y0, order, gm_map = pack_state_vector(ic)
    epoch_jd = ic["epoch_jd"]
    
    t_years = (times_jd_obs[-1] - epoch_jd) / 365.25
    result = propagate_from_state(y0, order, gm_map, t_years=t_years, A2=0.0, epoch_jd=epoch_jd)

    t_pred = result["times_jd"]
    pos_pred = result["asteroid_pos"]
    vel_pred = result["asteroid_vel"]

    # Calculo de semieje mayor observado
    a_obs_arr = np.array([semi_major_axis(pos_obs[i], vel_obs[i]) for i in range(len(times_jd_obs))])

    # Interpolacion de prediccion al grid observado
    a_pred_arr = np.zeros(len(times_jd_obs))
    for i, t in enumerate(times_jd_obs):
        pos_i = np.array([np.interp(t, t_pred, pos_pred[:, k]) for k in range(3)])
        vel_i = np.array([np.interp(t, t_pred, vel_pred[:, k]) for k in range(3)])
        a_pred_arr[i] = semi_major_axis(pos_i, vel_i)

    # Residuos
    epsilon = a_obs_arr - a_pred_arr
    epsilon_km = epsilon * 1.495978707e8
    times_years = (times_jd_obs - times_jd_obs[0]) / 365.25

    return ResidualSeries(
        times_jd=times_jd_obs, times_years=times_years,
        a_obs=a_obs_arr, a_pred=a_pred_arr,
        epsilon=epsilon, epsilon_km=epsilon_km,
        n_points=len(times_jd_obs),
        epoch_start=epoch_start, epoch_end=jd_to_iso(times_jd_obs[-1]),
        asteroid_id=int(asteroid_id),
    )

def simulate_short_arc(full_series: ResidualSeries, n_obs: int) -> ResidualSeries:
    """Simula un arco corto tomando N observaciones distribuidas en el tiempo."""
    if n_obs >= full_series.n_points:
        return full_series
    
    # Índices equiespaciados para mantener la línea de base temporal
    idx = np.linspace(0, full_series.n_points - 1, n_obs, dtype=int)
    
    return ResidualSeries(
        times_jd=full_series.times_jd[idx],
        times_years=full_series.times_years[idx],
        a_obs=full_series.a_obs[idx],
        a_pred=full_series.a_pred[idx],
        epsilon=full_series.epsilon[idx],
        epsilon_km=full_series.epsilon_km[idx],
        n_points=n_obs,
        epoch_start=full_series.epoch_start,
        epoch_end=full_series.epoch_end,
        asteroid_id=full_series.asteroid_id,
    )