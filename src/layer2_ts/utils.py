"""
utils.py
Utilidades Capa 2: conversiones, estadísticos, I/O y verificación.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from ..layer1_ode.utils import semi_major_axis, au_to_km, jd_to_iso
from ..layer1_ode.constants import GM_SOL_AU3_DAY2

def dadt_au_my_to_au_yr(dadt_au_my: float) -> float:
    return dadt_au_my / 1e6

def dadt_au_yr_to_au_my(dadt_au_yr: float) -> float:
    return dadt_au_yr * 1e6

def displacement_at_t(dadt_au_my: float, t_years: float) -> float:
    return dadt_au_my * 1e-6 * t_years

def displacement_km(dadt_au_my: float, t_years: float) -> float:
    return displacement_at_t(dadt_au_my, t_years) * au_to_km(1.0)

def describe_series(series) -> dict:
    eps, t = series.epsilon, series.times_years
    slope_au_my = np.polyfit(t, eps, 1)[0] * 1e6
    return {
        "n_points": series.n_points, "arc_years": float(series.times_years[-1]),
        "epsilon_mean_au": float(eps.mean()), "epsilon_std_au": float(eps.std()),
        "trend_slope_au_my": float(slope_au_my),
        "snr": abs(slope_au_my * series.times_years[-1] * 1e-6) / eps.std() if eps.std() > 0 else 0.0,
    }

def signal_to_noise_ratio(series) -> float:
    return float(describe_series(series)["snr"])

def save_series(series, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_dataframe().to_csv(path, index=False)
    # FIX: Uso de n_points en lugar de len(series)
    print(f"[HYPATIA L2] Serie guardada: {path} ({series.n_points} filas)")
    return path

def load_series(path: str | Path):
    from .residuals import ResidualSeries
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Serie no encontrada: {path}")
    df = pd.read_csv(path)
    required = ["times_jd", "times_years", "a_obs_au", "a_pred_au", "epsilon_au", "epsilon_km"]
    if missing := [c for c in required if c not in df.columns]:
        raise ValueError(f"CSV mal formado: columnas faltantes {missing}")
    return ResidualSeries(
        times_jd=df["times_jd"].values, times_years=df["times_years"].values,
        a_obs=df["a_obs_au"].values, a_pred=df["a_pred_au"].values,
        epsilon=df["epsilon_au"].values, epsilon_km=df["epsilon_km"].values,
        n_points=len(df), epoch_start=jd_to_iso(df["times_jd"].iloc[0]),
        epoch_end=jd_to_iso(df["times_jd"].iloc[-1]),
        asteroid_id=int(path.stem.split("_")[0]) if path.stem[0].isdigit() else 0,
    )

def verify_layer1_integration() -> bool:
    try:
        from ..layer1_ode import propagate_from_state, get_initial_conditions, pack_state_vector, semi_major_axis
        assert all(callable(f) for f in [propagate_from_state, get_initial_conditions, pack_state_vector, semi_major_axis])
        v_circ = np.sqrt(GM_SOL_AU3_DAY2 / 1.0)
        a = semi_major_axis(np.array([1.0, 0.0, 0.0]), np.array([0.0, v_circ, 0.0]))
        assert abs(a - 1.0) < 0.01, f"Semieje mayor Tierra erroneo: {a}"
        print("[HYPATIA L2] Integracion con Capa 1 verificada correctamente")
        return True
    except Exception as e:
        print(f"[HYPATIA L2] Error en integracion con Capa 1: {e}")
        return False

def check_residuals_quality(series) -> dict:
    """Valida calidad de serie antes de regresion."""
    warnings, errors, eps = [], [], series.epsilon
    
    if series.n_points < 5: 
        errors.append(f"Pocos puntos: {series.n_points} (min: 5)")
    if series.n_points < 10: 
        warnings.append(f"Pocos puntos ({series.n_points}): IC amplio")
        
    # FIX: Arco corto es esperado en experimentos de sensibilidad (N=5,10,20)
    # Solo se considera error fatal si es < 0.2 años (~2 meses)
    arc_years = series.times_years[-1]
    if arc_years < 0.2:
        errors.append(f"Arco inválido: {arc_years:.2f} años")
    elif arc_years < 1.0:
        warnings.append(f"Arco corto ({arc_years:.2f} años): IC amplio, válido para sensibilidad")
        
    outliers = np.abs(eps) > 0.01
    if np.any(outliers):
        warnings.append(f"{outliers.sum()} residuos > 0.01 AU")
        
    snr = signal_to_noise_ratio(series)
    if snr < 1.0: 
        warnings.append(f"SNR bajo ({snr:.2f})")
        
    passed = len(errors) == 0
    for w in warnings: print(f"[HYPATIA L2] Warning: {w}")
    for e in errors: print(f"[HYPATIA L2] Error: {e}")
    if passed and not warnings: 
        print(f"[HYPATIA L2] Serie verificada ({series.n_points} puntos, SNR={snr:.2f})")
    return {"passed": passed, "warnings": warnings, "errors": errors, "snr": snr}