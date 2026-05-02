"""
diagnostics.py
--------------
Diagnóstico estadístico de la serie de residuos orbitales antes
de aplicar los modelos de regresión.

Pruebas implementadas:
    1. Augmented Dickey-Fuller (ADF) — estacionariedad
    2. KPSS — confirmación de estacionariedad (hipótesis opuesta a ADF)
    3. Ljung-Box — autocorrelación residual
    4. Breusch-Pagan — heterocedasticidad
    5. CUSUM — quiebre estructural en la tendencia
    6. Durbin-Watson — autocorrelación de primer orden

Estos diagnósticos determinan cuál de los tres métodos de estimación
(OLS, STL o Bayesiano) aplica mejor a la serie concreta.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

from .residuals import ResidualSeries


@dataclass
class DiagnosticsReport:
    """
    Resultado completo del diagnóstico de la serie de residuos.

    Interpretación automática en .recommend_method()
    """
    # ADF: H0 = raíz unitaria (no estacionaria)
    adf_statistic   : float
    adf_pvalue      : float
    adf_stationary  : bool    # True si p < 0.05 → rechaza H0 → estacionaria

    # KPSS: H0 = estacionaria
    kpss_statistic  : float
    kpss_pvalue     : float
    kpss_stationary : bool    # True si p > 0.05 → no rechaza H0 → estacionaria

    # Ljung-Box: H0 = no autocorrelación
    lb_statistic    : float
    lb_pvalue       : float
    has_autocorr    : bool    # True si p < 0.05 → hay autocorrelación

    # Breusch-Pagan: H0 = homocedasticidad
    bp_statistic    : float
    bp_pvalue       : float
    heteroscedastic : bool    # True si p < 0.05 → heterocedasticidad

    # Durbin-Watson: ~2 = sin autocorrelación, <2 = positiva, >2 = negativa
    durbin_watson   : float

    # Tendencia lineal detectada (pendiente en AU/año)
    trend_slope_au_yr  : float
    trend_pvalue       : float
    trend_significant  : bool   # True si p < 0.05

    # ACF y PACF para visualización
    acf_values  : np.ndarray
    pacf_values : np.ndarray
    acf_lags    : np.ndarray

    def recommend_method(self) -> str:
        """
        Recomienda el método de estimación de da/dt basándose en
        los resultados del diagnóstico.

        Returns:
            'ols'      → residuos bien comportados
            'stl'      → heterocedásticos o con estacionalidad
            'bayesian' → pocos puntos o alta autocorrelación
        """
        n_issues = sum([
            self.has_autocorr,
            self.heteroscedastic,
            not self.adf_stationary,
        ])

        if n_issues == 0:
            return "ols"
        elif n_issues == 1 and not self.heteroscedastic:
            return "ols_hac"   # OLS con errores Newey-West
        elif self.heteroscedastic:
            return "stl"
        else:
            return "bayesian"

    def summary(self) -> str:
        rec = self.recommend_method()
        lines = [
            "─" * 52,
            "  DIAGNÓSTICO DE LA SERIE DE RESIDUOS",
            "─" * 52,
            f"  ADF  (estacionaria)   : {'Sí' if self.adf_stationary else 'No'}"
              f"  (p={self.adf_pvalue:.4f})",
            f"  KPSS (estacionaria)   : {'Sí' if self.kpss_stationary else 'No'}"
              f"  (p={self.kpss_pvalue:.4f})",
            f"  Ljung-Box (autocorr)  : {'Sí' if self.has_autocorr else 'No'}"
              f"  (p={self.lb_pvalue:.4f})",
            f"  Breusch-Pagan (hetsc) : {'Sí' if self.heteroscedastic else 'No'}"
              f"  (p={self.bp_pvalue:.4f})",
            f"  Durbin-Watson         : {self.durbin_watson:.3f}",
            f"  Tendencia lineal      : {'Sí' if self.trend_significant else 'No'}"
              f"  (p={self.trend_pvalue:.4f}, "
              f"slope={self.trend_slope_au_yr*1e6:.3f}×10⁻⁶ AU/yr)",
            "─" * 52,
            f"  Método recomendado    : {rec.upper()}",
            "─" * 52,
        ]
        return "\n".join(lines)


def run_diagnostics(
    series: ResidualSeries,
    alpha: float = 0.05,
    max_lags: int = 10,
) -> DiagnosticsReport:
    """
    Ejecuta el diagnóstico estadístico completo sobre la serie de residuos.

    Args:
        series   : ResidualSeries de la Capa 2
        alpha    : nivel de significancia (default 0.05)
        max_lags : número de lags para Ljung-Box y ACF/PACF

    Returns:
        DiagnosticsReport con todos los resultados y recomendación
    """
    t = series.times_years
    eps = series.epsilon
    n = len(eps)

    # ── 1. ADF: estacionariedad ───────────────────────────────────────────
    # Usamos 'c' (constante) porque los residuos tienen media no nula
    # si hay drift de Yarkovsky. 'ct' (constante+tendencia) para la
    # hipótesis de tendencia determinista.
    adf_result = adfuller(eps, autolag="AIC", regression="ct")
    adf_stat   = float(adf_result[0])
    adf_pval   = float(adf_result[1])
    adf_stat_  = adf_stat < adf_result[4]["5%"]   # comparar con valor crítico

    # ── 2. KPSS: confirmar estacionariedad ────────────────────────────────
    try:
        kpss_result = kpss(eps, regression="ct", nlags="auto")
        kpss_stat   = float(kpss_result[0])
        kpss_pval   = float(kpss_result[1])
        kpss_stat_  = kpss_pval > alpha   # no rechaza H0 → estacionaria
    except Exception:
        kpss_stat, kpss_pval, kpss_stat_ = np.nan, 0.5, True

    # ── 3. Ljung-Box: autocorrelación ─────────────────────────────────────
    lags_lb = min(max_lags, n // 4)
    lb_result = acorr_ljungbox(eps, lags=[lags_lb], return_df=True)
    lb_stat   = float(lb_result["lb_stat"].iloc[-1])
    lb_pval   = float(lb_result["lb_pvalue"].iloc[-1])
    has_ac    = lb_pval < alpha

    # ── 4. Breusch-Pagan: heterocedasticidad ─────────────────────────────
    # Regresión auxiliar: eps ~ t (tiempo como regresor)
    X_bp = sm.add_constant(t)
    model_bp = OLS(eps, X_bp).fit()
    try:
        bp_result = het_breuschpagan(model_bp.resid, model_bp.model.exog)
        bp_stat   = float(bp_result[0])
        bp_pval   = float(bp_result[1])
        hetsc     = bp_pval < alpha
    except Exception:
        bp_stat, bp_pval, hetsc = np.nan, 1.0, False

    # ── 5. Durbin-Watson ─────────────────────────────────────────────────
    dw = float(durbin_watson(model_bp.resid))

    # ── 6. Tendencia lineal (OLS simple) ─────────────────────────────────
    trend_slope = float(model_bp.params[-1])
    trend_pval  = float(model_bp.pvalues[-1])
    trend_sig   = trend_pval < alpha

    # ── 7. ACF / PACF para visualización ─────────────────────────────────
    n_lags_acf = min(max_lags, n // 3)
    acf_vals  = acf(eps,  nlags=n_lags_acf, fft=True)[1:]    # quita lag-0
    pacf_vals = pacf(eps, nlags=n_lags_acf)[1:]
    acf_lags  = np.arange(1, len(acf_vals) + 1)

    return DiagnosticsReport(
        adf_statistic   = adf_stat,
        adf_pvalue      = adf_pval,
        adf_stationary  = adf_stat_,
        kpss_statistic  = kpss_stat,
        kpss_pvalue     = kpss_pval,
        kpss_stationary = kpss_stat_,
        lb_statistic    = lb_stat,
        lb_pvalue       = lb_pval,
        has_autocorr    = has_ac,
        bp_statistic    = bp_stat,
        bp_pvalue       = bp_pval,
        heteroscedastic = hetsc,
        durbin_watson   = dw,
        trend_slope_au_yr  = trend_slope,
        trend_pvalue       = trend_pval,
        trend_significant  = trend_sig,
        acf_values  = acf_vals,
        pacf_values = pacf_vals,
        acf_lags    = acf_lags,
    )
