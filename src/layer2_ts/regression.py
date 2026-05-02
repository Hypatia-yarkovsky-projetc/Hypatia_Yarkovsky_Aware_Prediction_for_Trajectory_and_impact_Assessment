"""
regression.py
-------------
Estimación del parámetro de Yarkovsky da/dt desde la serie de residuos
orbitales mediante tres métodos estadísticos en orden de sofisticación.

Método 1 — OLS
    ε(t) = β₀ + β₁·t + η(t)
    da/dt = β₁  [AU/año → convertir a AU/My]
    Aplica cuando: residuos bien comportados (homoescedásticos, sin autocorr)

Método 2 — OLS con errores Newey-West (HAC)
    Igual que OLS pero con errores estándar robustos a autocorrelación
    y heterocedasticidad (HAC: heteroscedasticity and autocorrelation consistent).
    Aplica cuando: los diagnósticos detectan autocorrelación moderada.

Método 3 — Descomposición STL + regresión sobre tendencia
    STL separa ε(t) = T(t) + S(t) + R(t)
    Luego regresión OLS sobre T(t) únicamente, eliminando ruido y estacionalidad.
    Aplica cuando: heterocedasticidad o estacionalidad detectada.

Todos los métodos producen:
    - Estimación puntual de da/dt [AU/My]
    - Intervalo de confianza al 95%
    - Métricas de ajuste (R², RMSE, AIC)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.stattools import durbin_watson

from .residuals import ResidualSeries


# ── Constante de conversión AU/año → AU/My ───────────────────────────────
AU_YR_TO_AU_MY = 1e6   # 1 AU/año = 1×10⁶ AU/My  → NO, al revés:
# 1 AU/My = 1e-6 AU/año  → da/dt [AU/año] * 1e6 = da/dt [AU/My]
# Corrección:
# slope [AU/año] → dadt [AU/My] = slope * 1e6
# Porque 1 My = 1e6 años → dadt [AU/My] = slope [AU/año] × 1e6


@dataclass
class RegressionResult:
    """
    Resultado de la estimación de da/dt por cualquier método.

    Todos los valores de da/dt en AU/My (unidad estándar del campo).
    """
    method       : str    # 'ols', 'ols_hac', 'stl'
    dadt_au_my   : float  # estimación puntual
    ci_lower     : float  # límite inferior IC 95%
    ci_upper     : float  # límite superior IC 95%
    std_error    : float  # error estándar de la estimación
    r_squared    : float  # coeficiente de determinación
    rmse_au      : float  # RMSE del ajuste en AU
    aic          : float  # criterio de información de Akaike
    n_points     : int    # número de observaciones usadas
    dw_statistic : float  # Durbin-Watson del modelo

    # Vectores del ajuste para visualización
    t_fit        : np.ndarray = field(default_factory=lambda: np.array([]))
    trend_fit    : np.ndarray = field(default_factory=lambda: np.array([]))
    residuals_fit: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    @property
    def is_significant(self) -> bool:
        """True si el IC 95% no contiene el cero."""
        return not (self.ci_lower <= 0 <= self.ci_upper)

    def summary(self) -> str:
        sig = "✓ significativo" if self.is_significant else "✗ no significativo"
        lines = [
            f"  Método        : {self.method.upper()}",
            f"  da/dt         : {self.dadt_au_my:+.4f} AU/My  {sig}",
            f"  IC 95%        : [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}] AU/My",
            f"  Error estd    : {self.std_error:.4f} AU/My",
            f"  R²            : {self.r_squared:.4f}",
            f"  RMSE          : {self.rmse_au*1e6:.4f} ×10⁻⁶ AU",
            f"  Puntos        : {self.n_points}",
            f"  Durbin-Watson : {self.dw_statistic:.3f}",
        ]
        return "\n".join(lines)


# ── MÉTODO 1: OLS ─────────────────────────────────────────────────────────

def estimate_ols(series: ResidualSeries, alpha: float = 0.05) -> RegressionResult:
    """
    Estimación OLS de da/dt: regresión lineal de los residuos sobre
    el tiempo.

        ε(t) = β₀ + β₁·t + η(t)
        da/dt = β₁ × 10⁶  [AU/año → AU/My]

    Errores estándar estándar (asumen η ~ IID). Para series con
    autocorrelación, usar estimate_ols_hac().

    Args:
        series : ResidualSeries de build_residual_series()
        alpha  : nivel de significancia para el IC (default: 0.05 → IC 95%)

    Returns:
        RegressionResult con la estimación y métricas de ajuste
    """
    t   = series.times_years
    eps = series.epsilon

    X = sm.add_constant(t)
    model = sm.OLS(eps, X).fit()

    # Extraer coeficientes
    slope     = float(np.asarray(model.params)[1])      # β₁ en AU/año
    intercept = float(model.params[0])
    se_slope  = float(np.asarray(model.bse)[1])         # error estándar clásico

    ci = model.conf_int(alpha=alpha)
    ci_lo = float(ci[1, 0])
    ci_hi = float(ci[1, 1])

    fitted    = np.asarray(model.fittedvalues)
    residuals = np.asarray(model.resid)
    dw        = float(durbin_watson(residuals))

    return RegressionResult(
        method       = "ols",
        dadt_au_my   = slope * 1e6,
        ci_lower     = ci_lo * 1e6,
        ci_upper     = ci_hi * 1e6,
        std_error    = se_slope * 1e6,
        r_squared    = float(model.rsquared),
        rmse_au      = float(np.sqrt(np.mean(residuals**2))),
        aic          = float(model.aic),
        n_points     = len(t),
        dw_statistic = dw,
        t_fit        = t,
        trend_fit    = fitted,
        residuals_fit= residuals,
    )


# ── MÉTODO 2: OLS con errores Newey-West (HAC) ────────────────────────────

def estimate_ols_hac(
    series: ResidualSeries,
    alpha : float = 0.05,
    n_lags: Optional[int] = None,
) -> RegressionResult:
    """
    OLS con errores estándar HAC (Newey-West): robustos a autocorrelación
    y heterocedasticidad de forma no paramétrica.

    Es la versión correcta de OLS cuando los diagnósticos detectan
    autocorrelación en los residuos (Ljung-Box significativo o DW < 1.5).

    El número de lags de Newey-West se selecciona automáticamente con
    la regla de Andrews (1991): floor(4·(n/100)^(2/9)) si n_lags=None.

    Args:
        series : ResidualSeries
        alpha  : nivel de significancia
        n_lags : número de lags para Newey-West (None = automático)

    Returns:
        RegressionResult con errores HAC
    """
    t   = series.times_years
    eps = series.epsilon
    n   = len(t)

    # Selección automática de lags (Andrews 1991)
    if n_lags is None:
        n_lags = max(1, int(np.floor(4 * (n / 100) ** (2/9))))

    X = sm.add_constant(t)
    model_ols = sm.OLS(eps, X).fit()

    # Re-ajustar con covarianza Newey-West
    model_hac = model_ols.get_robustcov_results(
        cov_type="HAC", maxlags=n_lags, use_correction=True
    )

    slope    = float(float(np.asarray(model_hac.params)[1]))
    se_hac   = float(float(np.asarray(model_hac.bse)[1]))

    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(1 - alpha/2, df=n - 2)
    ci_lo  = slope - t_crit * se_hac
    ci_hi  = slope + t_crit * se_hac

    residuals = np.asarray(model_ols.resid)
    dw        = float(durbin_watson(residuals))

    return RegressionResult(
        method       = "ols_hac",
        dadt_au_my   = slope * 1e6,
        ci_lower     = ci_lo * 1e6,
        ci_upper     = ci_hi * 1e6,
        std_error    = se_hac * 1e6,
        r_squared    = float(model_ols.rsquared),
        rmse_au      = float(np.sqrt(np.mean(residuals**2))),
        aic          = float(model_ols.aic),
        n_points     = n,
        dw_statistic = dw,
        t_fit        = t,
        trend_fit    = np.asarray(model_ols.fittedvalues),
        residuals_fit= residuals,
    )


# ── MÉTODO 3: STL + regresión sobre tendencia ────────────────────────────

def estimate_stl(
    series     : ResidualSeries,
    alpha      : float = 0.05,
    period     : Optional[int] = None,
    robust     : bool = True,
    seasonal_deg: int = 1,
    trend_deg  : int = 1,
) -> RegressionResult:
    """
    Descomposición STL (Seasonal-Trend decomposition using Loess) seguida
    de regresión OLS sobre el componente de tendencia.

        ε(t) = T(t) + S(t) + R(t)
        da/dt estimado desde regresión lineal de T(t)

    STL es más robusto que OLS ante outliers, gaps en los datos y
    variación estacional (que en series orbitales corresponde al
    período de la ventana de observación, típicamente ~1 año).

    Args:
        series      : ResidualSeries
        alpha       : nivel de significancia
        period      : período estacional en puntos (None = auto-detectar)
        robust      : usar versión robusta de STL (resistente a outliers)
        seasonal_deg: grado del polinomio estacional (1 = lineal)
        trend_deg   : grado del polinomio de tendencia

    Returns:
        RegressionResult con da/dt estimado desde T(t)
    """
    t   = series.times_years
    eps = series.epsilon
    n   = len(eps)

    # Auto-detectar período: si las obs son cada 30d, ~12 por año
    if period is None:
        dt_median_days = np.median(np.diff(series.times_jd))
        period = max(2, int(round(365.25 / dt_median_days)))

    # STL requiere al menos 2 períodos completos
    if n < 2 * period:
        # Con pocos datos, caer a OLS-HAC
        print(f"[HYPATIA L2] STL: n={n} < 2×período={2*period}, usando OLS-HAC")
        return estimate_ols_hac(series, alpha)

    # Crear índice temporal regular para STL
    idx = pd.RangeIndex(n)
    ts  = pd.Series(eps, index=idx)

    stl = STL(
        ts,
        period       = period,
        robust       = robust,
        seasonal_deg = seasonal_deg,
        trend_deg    = trend_deg,
    )
    result = stl.fit()

    trend_component = result.trend.values    # T(t): tendencia suavizada

    # OLS sobre la tendencia (sin ruido ni estacionalidad)
    X     = sm.add_constant(t)
    model = sm.OLS(trend_component, X).fit()

    slope    = float(np.asarray(model.params)[1])
    se_slope = float(np.asarray(model.bse)[1])

    ci = model.conf_int(alpha=alpha)
    ci_lo = float(ci[1, 0])
    ci_hi = float(ci[1, 1])

    # Residuos del STL como diagnóstico de ajuste
    stl_residuals = result.resid.values
    dw = float(durbin_watson(stl_residuals))

    return RegressionResult(
        method       = "stl",
        dadt_au_my   = slope * 1e6,
        ci_lower     = ci_lo * 1e6,
        ci_upper     = ci_hi * 1e6,
        std_error    = se_slope * 1e6,
        r_squared    = float(model.rsquared),
        rmse_au      = float(np.sqrt(np.mean(stl_residuals**2))),
        aic          = float(model.aic),
        n_points     = n,
        dw_statistic = dw,
        t_fit        = t,
        trend_fit    = trend_component,
        residuals_fit= stl_residuals,
    )


# ── Función de alto nivel: correr todos los métodos ──────────────────────

def estimate_dadt_all_methods(
    series : ResidualSeries,
    alpha  : float = 0.05,
) -> dict[str, RegressionResult]:
    """
    Corre los tres métodos sobre la misma serie y devuelve un diccionario
    con los resultados para comparación.

    Args:
        series : ResidualSeries
        alpha  : nivel de significancia

    Returns:
        dict {'ols': RegressionResult, 'ols_hac': ..., 'stl': ...}
    """
    results = {
        "ols"    : estimate_ols(series, alpha),
        "ols_hac": estimate_ols_hac(series, alpha),
        "stl"    : estimate_stl(series, alpha),
    }

    print(f"\n[HYPATIA L2] Estimaciones de da/dt — {series.n_points} observaciones:")
    print(f"  {'Método':<12} {'da/dt (AU/My)':>14} {'IC 95%':>28} {'R²':>6}")
    print(f"  {'─'*64}")
    for name, r in results.items():
        print(f"  {name:<12} {r.dadt_au_my:>+14.4f}"
              f"  [{r.ci_lower:>+8.4f}, {r.ci_upper:>+8.4f}]"
              f"  {r.r_squared:>6.3f}")

    return results


def sensitivity_analysis(
    full_series  : ResidualSeries,
    n_obs_list   : list[int] = [5, 10, 20, 30, 50],
    method       : str = "ols_hac",
    alpha        : float = 0.05,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad: cómo varía la estimación de da/dt al
    cambiar el número de observaciones disponibles.

    Este es el experimento de arco corto: simula el escenario de
    objeto recién descubierto con N observaciones.

    Args:
        full_series : ResidualSeries completa
        n_obs_list  : lista de tamaños de arco a probar
        method      : 'ols', 'ols_hac' o 'stl'
        alpha       : nivel de significancia

    Returns:
        DataFrame con columnas: n_obs, dadt, ci_lower, ci_upper, std_err, r2
    """
    from .residuals import simulate_short_arc

    estimators = {"ols": estimate_ols, "ols_hac": estimate_ols_hac, "stl": estimate_stl}
    estimator = estimators.get(method, estimate_ols_hac)

    rows = []
    for n in n_obs_list:
        if n > full_series.n_points:
            continue
        short = simulate_short_arc(full_series, n)
        try:
            res = estimator(short, alpha)
            rows.append({
                "n_obs"    : n,
                "arc_years": float(short.times_years[-1]),
                "dadt"     : res.dadt_au_my,
                "ci_lower" : res.ci_lower,
                "ci_upper" : res.ci_upper,
                "ci_width" : res.ci_width,
                "std_err"  : res.std_error,
                "r2"       : res.r_squared,
            })
        except Exception as e:
            print(f"  [WARN] n={n}: {e}")

    df = pd.DataFrame(rows)
    print(f"\n[HYPATIA L2] Análisis de sensibilidad — método {method.upper()}:")
    print(df.to_string(index=False, float_format="{:+.4f}".format))
    return df


Optional = type(None) | type  # Re-declarar para compatibilidad
from typing import Optional
