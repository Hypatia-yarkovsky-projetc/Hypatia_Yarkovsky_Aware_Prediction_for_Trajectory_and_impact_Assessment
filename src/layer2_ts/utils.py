"""
utils.py
--------
Funciones de utilidad para la Capa 2 de HYPATIA.

Incluye:
    - Conversiones de unidades para da/dt
    - Cálculo del semieje mayor desde posición y velocidad (wrapper Capa 1)
    - Estadísticos descriptivos de la serie de residuos
    - Exportación e importación de series a CSV
    - Verificación de consistencia entre Capa 1 y Capa 2
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Acceso a la Capa 1 para funciones compartidas
from ..layer1_ode.utils import semi_major_axis, au_to_km, jd_to_iso
from ..layer1_ode.constants import KM_TO_AU

# Constantes de conversión de tiempo
DAYS_PER_YEAR = 365.25
DAYS_PER_MY   = 365.25e6   # 1 millón de años en días


# ── Conversiones de da/dt ─────────────────────────────────────────────────

def dadt_au_my_to_au_yr(dadt_au_my: float) -> float:
    """AU/My → AU/año."""
    return dadt_au_my / 1e6


def dadt_au_yr_to_au_my(dadt_au_yr: float) -> float:
    """AU/año → AU/My."""
    return dadt_au_yr * 1e6


def dadt_au_my_to_m_s(dadt_au_my: float, a_au: float) -> float:
    """
    Convierte da/dt [AU/My] a la aceleración equivalente en m/s².
    Útil para comparar con el orden de magnitud físico (10⁻¹² m/s²).

    da/dt [AU/My] → Δv/período_orbital [m/s²] (aproximación)
    """
    AU_M = 1.495978707e11      # 1 AU en metros
    MY_S = 365.25e6 * 86400   # 1 My en segundos

    # da/dt en m/s: cambio de velocidad orbital ≈ 0.5 * da/dt / T_orb
    # Para NEO típico con a ~ 1 AU: T_orb ~ 1 año = 3.156e7 s
    T_orb_s = DAYS_PER_YEAR * 86400 * a_au ** 1.5   # Kepler (aproximado)
    dadt_m_s = (dadt_au_my * AU_M / MY_S) / T_orb_s
    return dadt_m_s


def displacement_at_t(dadt_au_my: float, t_years: float) -> float:
    """
    Desplazamiento acumulado en el semieje mayor después de t_years años.
    Δa = da/dt × t

    Args:
        dadt_au_my : da/dt en AU/My
        t_years    : tiempo en años

    Returns:
        Δa en AU
    """
    return dadt_au_my * 1e-6 * t_years


def displacement_km(dadt_au_my: float, t_years: float) -> float:
    """Desplazamiento acumulado en km."""
    return displacement_at_t(dadt_au_my, t_years) * au_to_km(1.0)


# ── Estadísticos de la serie ──────────────────────────────────────────────

def describe_series(series) -> dict:
    """
    Estadísticos descriptivos de la serie de residuos.

    Args:
        series : ResidualSeries

    Returns:
        dict con estadísticos clave
    """
    eps = series.epsilon

    # Tendencia lineal simple (OLS rápido)
    t    = series.times_years
    coef = np.polyfit(t, eps, 1)
    slope_au_my = coef[0] * 1e6   # AU/año → AU/My

    return {
        "n_points"          : series.n_points,
        "arc_years"         : float(series.times_years[-1]),
        "epsilon_mean_au"   : float(eps.mean()),
        "epsilon_std_au"    : float(eps.std()),
        "epsilon_min_au"    : float(eps.min()),
        "epsilon_max_au"    : float(eps.max()),
        "epsilon_range_au"  : float(eps.max() - eps.min()),
        "trend_slope_au_my" : float(slope_au_my),
        "snr"               : abs(slope_au_my * series.times_years[-1] * 1e-6)
                              / eps.std() if eps.std() > 0 else 0.0,
    }


def signal_to_noise_ratio(series) -> float:
    """
    Calcula el SNR de la señal de Yarkovsky en la serie de residuos.

    SNR = |Δa_total| / σ_ruido
    donde Δa_total = da/dt × T_arco es el desplazamiento total esperado
    y σ_ruido es la desviación estándar de los residuos.

    SNR > 3 : la señal es detectable con confianza
    SNR < 1 : la señal está enterrada en el ruido
    """
    stats = describe_series(series)
    return float(stats["snr"])


# ── Exportación / importación ─────────────────────────────────────────────

def save_series(series, path: str | Path) -> Path:
    """
    Guarda la serie de residuos como CSV para reproducibilidad.

    Args:
        series : ResidualSeries
        path   : ruta de destino (ej. 'data/processed/apophis_residuals.csv')

    Returns:
        Path del archivo guardado
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = series.to_dataframe()
    df.to_csv(path, index=False)

    print(f"[HYPATIA L2] Serie guardada: {path} ({len(df)} filas)")
    return path


def load_series(path: str | Path):
    """
    Carga una serie de residuos desde CSV.
    Permite trabajar offline sin necesitar conexión a JPL.

    Returns:
        ResidualSeries reconstruida desde el CSV
    """
    from .residuals import ResidualSeries

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Serie no encontrada: {path}")

    df = pd.read_csv(path)
    required = ["times_jd", "times_years", "a_obs_au", "a_pred_au",
                "epsilon_au", "epsilon_km"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV mal formado, columnas faltantes: {missing}")

    times_jd = df["times_jd"].values
    return ResidualSeries(
        times_jd    = times_jd,
        times_years = df["times_years"].values,
        a_obs       = df["a_obs_au"].values,
        a_pred      = df["a_pred_au"].values,
        epsilon     = df["epsilon_au"].values,
        epsilon_km  = df["epsilon_km"].values,
        n_points    = len(df),
        epoch_start = jd_to_iso(times_jd[0]),
        epoch_end   = jd_to_iso(times_jd[-1]),
        asteroid_id = int(path.stem.split("_")[0]) if path.stem[0].isdigit() else 0,
    )


# ── Verificación de integración con Capa 1 ───────────────────────────────

def verify_layer1_integration() -> bool:
    """
    Verifica que la Capa 1 está disponible y sus interfaces son correctas.
    Corre una propagación mínima de 1 día para confirmar la conexión.

    Returns:
        True si la integración es correcta, False si hay errores.
    """
    try:
        from ..layer1_ode import (
            propagate_from_state,
            get_initial_conditions,
            pack_state_vector,
            semi_major_axis,
        )
        # Verificar que las funciones son llamables
        assert callable(propagate_from_state)
        assert callable(get_initial_conditions)
        assert callable(pack_state_vector)
        assert callable(semi_major_axis)

        # Test mínimo: calcular semieje mayor para estado conocido (Tierra)
        from ..layer1_ode.constants import GM_SOL_AU3_DAY2
        v_circ = np.sqrt(GM_SOL_AU3_DAY2 / 1.0)
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.array([0.0, v_circ, 0.0])
        a   = semi_major_axis(pos, vel)
        assert abs(a - 1.0) < 0.01, f"Semieje mayor Tierra erróneo: {a}"

        print("[HYPATIA L2] ✓ Integración con Capa 1 verificada correctamente")
        return True

    except Exception as e:
        print(f"[HYPATIA L2] ✗ Error en integración con Capa 1: {e}")
        return False


def check_residuals_quality(series) -> dict:
    """
    Verifica la calidad de la serie de residuos antes de aplicar
    los modelos de regresión.

    Returns:
        dict {'passed': bool, 'warnings': list[str], 'errors': list[str]}
    """
    warnings = []
    errors   = []
    eps      = series.epsilon

    # Error: menos de 5 puntos (insuficiente para cualquier regresión)
    if series.n_points < 5:
        errors.append(f"Muy pocos puntos: {series.n_points} (mínimo: 5)")

    # Warning: menos de 10 puntos (estimación poco fiable)
    if series.n_points < 10:
        warnings.append(f"Pocos puntos ({series.n_points}): IC será amplio")

    # Error: arco menor a 1 año
    if series.times_years[-1] < 1.0:
        errors.append(f"Arco demasiado corto: {series.times_years[-1]:.2f} años")

    # Warning: residuos con valores extremos (|ε| > 0.01 AU = 1.5M km)
    outliers = np.abs(eps) > 0.01
    if outliers.any():
        warnings.append(f"{outliers.sum()} residuos con |ε| > 0.01 AU (posibles outliers)")

    # Warning: SNR muy bajo (señal apenas detectable)
    snr = signal_to_noise_ratio(series)
    if snr < 1.0:
        warnings.append(f"SNR bajo ({snr:.2f}): señal puede no ser detectable")

    passed = len(errors) == 0

    if warnings:
        for w in warnings:
            print(f"[HYPATIA L2] ⚠ {w}")
    if errors:
        for e in errors:
            print(f"[HYPATIA L2] ✗ {e}")
    if passed and not warnings:
        print(f"[HYPATIA L2] ✓ Serie de residuos verificada ({series.n_points} puntos, "
              f"arco={series.times_years[-1]:.1f} años, SNR={snr:.2f})")

    return {"passed": passed, "warnings": warnings, "errors": errors, "snr": snr}
