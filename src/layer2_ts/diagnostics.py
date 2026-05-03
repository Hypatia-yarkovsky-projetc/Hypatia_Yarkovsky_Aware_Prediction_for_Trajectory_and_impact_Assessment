"""
diagnostics.py
Diagnóstico estadístico de la serie de residuos.
Pruebas: ADF, KPSS, Ljung-Box, Breusch-Pagan, Durbin-Watson.
"""
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from .residuals import ResidualSeries

@dataclass
class DiagnosticsReport:
    """Reporte de diagnóstico estadístico."""
    adf_statistic: float; adf_pvalue: float; adf_stationary: bool
    kpss_statistic: float; kpss_pvalue: float; kpss_stationary: bool
    lb_statistic: float; lb_pvalue: float; has_autocorr: bool
    bp_statistic: float; bp_pvalue: float; heteroscedastic: bool
    durbin_watson: float
    trend_slope_au_yr: float; trend_pvalue: float; trend_significant: bool
    acf_values: np.ndarray; pacf_values: np.ndarray; acf_lags: np.ndarray

    def recommend_method(self) -> str:
        """Recomienda método basado en diagnóstico."""
        issues = sum([self.has_autocorr, self.heteroscedastic, not self.adf_stationary])
        if issues == 0: return "ols"
        if issues == 1 and not self.heteroscedastic: return "ols_hac"
        return "stl"

    def summary(self) -> str:
        rec = self.recommend_method()
        return (
            f"Diagnóstico: ADF({'Sí' if self.adf_stationary else 'No'}) | "
            f"Autocorr({'Sí' if self.has_autocorr else 'No'}) | "
            f"Hetsc({'Sí' if self.heteroscedastic else 'No'}) | "
            f"Método recomendado: {rec.upper()}"
        )

def run_diagnostics(series: ResidualSeries, alpha: float = 0.05, max_lags: int = 10) -> DiagnosticsReport:
    """Ejecuta suite de diagnósticos."""
    t, eps, n = series.times_years, series.epsilon, len(series.epsilon)
    
    # 1. ADF
    adf = adfuller(eps, autolag="AIC", regression="ct")
    adf_stat = float(adf[0])
    
    # 2. KPSS
    try:
        kpss_res = kpss(eps, regression="ct", nlags="auto")
        kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
    except Exception:
        kpss_stat, kpss_p = np.nan, 1.0

    # 3. Ljung-Box
    lags = min(max_lags, n // 4)
    lb_res = acorr_ljungbox(eps, lags=[lags], return_df=True)
    lb_stat, lb_p = float(lb_res["lb_stat"].iloc[-1]), float(lb_res["lb_pvalue"].iloc[-1])

    # 4. Breusch-Pagan
    X = sm.add_constant(t)
    model_bp = OLS(eps, X).fit()
    try:
        bp_res = het_breuschpagan(model_bp.resid, model_bp.model.exog)
        bp_stat, bp_p = float(bp_res[0]), float(bp_res[1])
    except Exception:
        bp_stat, bp_p = np.nan, 1.0

    # 5. Tendencia y DW
    trend_slope = float(model_bp.params[-1])
    
    return DiagnosticsReport(
        adf_statistic=adf_stat, adf_pvalue=float(adf[1]), adf_stationary=adf[1] < alpha,
        kpss_statistic=kpss_stat, kpss_pvalue=kpss_p, kpss_stationary=kpss_p > alpha,
        lb_statistic=lb_stat, lb_pvalue=lb_p, has_autocorr=lb_p < alpha,
        bp_statistic=bp_stat, bp_pvalue=bp_p, heteroscedastic=bp_p < alpha,
        durbin_watson=float(durbin_watson(model_bp.resid)),
        trend_slope_au_yr=trend_slope, trend_pvalue=float(model_bp.pvalues[-1]), trend_significant=model_bp.pvalues[-1] < alpha,
        acf_values=acf(eps, nlags=min(max_lags, n//3), fft=True)[1:],
        pacf_values=pacf(eps, nlags=min(max_lags, n//3))[1:],
        acf_lags=np.arange(1, min(max_lags, n//3) + 1),
    )