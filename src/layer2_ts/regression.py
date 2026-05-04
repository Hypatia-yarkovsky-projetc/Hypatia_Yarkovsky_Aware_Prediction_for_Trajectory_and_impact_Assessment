"""
regression.py
Estimación del parámetro de Yarkovsky da/dt desde la serie de residuos
orbitales mediante tres métodos estadísticos.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import t as t_dist
from .residuals import ResidualSeries

@dataclass
class RegressionResult:
    """Resultado de la estimación de da/dt."""
    method       : str
    dadt_au_my   : float
    ci_lower     : float
    ci_upper     : float
    std_error    : float
    r_squared    : float
    rmse_au      : float
    aic          : float
    n_points     : int
    dw_statistic : float
    t_fit        : np.ndarray = field(default_factory=lambda: np.array([]))
    trend_fit    : np.ndarray = field(default_factory=lambda: np.array([]))
    residuals_fit: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    @property
    def is_significant(self) -> bool:
        return not (self.ci_lower <= 0 <= self.ci_upper)

    def summary(self) -> str:
        sig = "significativo" if self.is_significant else "no significativo"
        return (
            f"Método: {self.method.upper()} | "
            f"da/dt: {self.dadt_au_my:+.4f} AU/My ({sig}) | "
            f"IC 95%: [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}] | "
            f"R²: {self.r_squared:.3f} | Puntos: {self.n_points}"
        )

def estimate_ols(series: ResidualSeries, alpha: float = 0.05) -> RegressionResult:
    """OLS básico."""
    t, eps = series.times_years, series.epsilon
    X = sm.add_constant(t)
    model = sm.OLS(eps, X).fit()

    params = np.asarray(model.params)
    bse = np.asarray(model.bse)
    slope = float(params[1])
    se_slope = float(bse[1])

    ci = np.asarray(model.conf_int(alpha=alpha))
    ci_lo = float(ci[1, 0])
    ci_hi = float(ci[1, 1])

    residuals = np.asarray(model.resid)
    dw = float(durbin_watson(residuals))

    return RegressionResult(
        method="ols", dadt_au_my=slope * 1e6,
        ci_lower=ci_lo * 1e6, ci_upper=ci_hi * 1e6,
        std_error=se_slope * 1e6,
        r_squared=float(model.rsquared),
        rmse_au=float(np.sqrt(np.mean(residuals**2))),
        aic=float(model.aic), n_points=len(t),
        dw_statistic=dw,
        t_fit=t, trend_fit=np.asarray(model.fittedvalues), residuals_fit=residuals,
    )

def estimate_ols_hac(series: ResidualSeries, alpha: float = 0.05, n_lags: Optional[int] = None) -> RegressionResult:
    """OLS con errores HAC (Newey-West)."""
    t, eps, n = series.times_years, series.epsilon, len(series.epsilon)
    if n_lags is None:
        n_lags = max(1, int(np.floor(4 * (n / 100) ** (2/9))))

    X = sm.add_constant(t)
    model_ols = sm.OLS(eps, X).fit()
    model_hac = model_ols.get_robustcov_results(cov_type="HAC", maxlags=n_lags, use_correction=True)

    params = np.asarray(model_hac.params)
    bse = np.asarray(model_hac.bse)
    slope = float(params[1])
    se_hac = float(bse[1])

    t_crit = t_dist.ppf(1 - alpha/2, df=n - 2)
    ci_lo = slope - t_crit * se_hac
    ci_hi = slope + t_crit * se_hac

    residuals = np.asarray(model_ols.resid)
    dw = float(durbin_watson(residuals))

    return RegressionResult(
        method="ols_hac", dadt_au_my=slope * 1e6,
        ci_lower=ci_lo * 1e6, ci_upper=ci_hi * 1e6,
        std_error=se_hac * 1e6,
        r_squared=float(model_ols.rsquared),
        rmse_au=float(np.sqrt(np.mean(residuals**2))),
        aic=float(model_ols.aic), n_points=n,
        dw_statistic=dw,
        t_fit=t, trend_fit=np.asarray(model_ols.fittedvalues), residuals_fit=residuals,
    )

def estimate_stl(
    series: ResidualSeries,
    alpha: float = 0.05,
    period: Optional[int] = None,
) -> RegressionResult:
    """Estimación STL + regresión sobre tendencia."""
    t, eps, n = series.times_years, series.epsilon, len(series.epsilon)
    
    if period is None:
        dt_med = np.median(np.diff(series.times_jd))
        period = max(2, int(round(365.25 / dt_med)))

    if n < 2 * period:
        return estimate_ols_hac(series, alpha)

    idx = pd.RangeIndex(n)
    ts = pd.Series(eps, index=idx)
    stl_res = STL(ts, period=period, robust=True).fit()
    
    X = sm.add_constant(t)
    model = sm.OLS(stl_res.trend, X).fit()
    
    # FIX: Uso de .iloc para acceso por posición, no por etiqueta
    slope = float(model.params.iloc[1])
    se_slope = float(model.bse.iloc[1])

    ci = model.conf_int(alpha=alpha)
    ci_lo = float(ci.iloc[1, 0])
    ci_hi = float(ci.iloc[1, 1])

    residuals = stl_res.resid

    return RegressionResult(
        method="stl", dadt_au_my=slope * 1e6,
        ci_lower=ci_lo * 1e6, ci_upper=ci_hi * 1e6,
        std_error=se_slope * 1e6,
        r_squared=float(model.rsquared),
        rmse_au=float(np.sqrt(np.mean(residuals**2))),
        aic=float(model.aic), n_points=n,
        dw_statistic=float(durbin_watson(residuals)),
        t_fit=t, trend_fit=stl_res.trend, residuals_fit=residuals,
    )

def estimate_dadt_all_methods(series: ResidualSeries, alpha: float = 0.05) -> dict:
    """Ejecuta los tres métodos."""
    return {
        "ols": estimate_ols(series, alpha),
        "ols_hac": estimate_ols_hac(series, alpha),
        "stl": estimate_stl(series, alpha),
    }

def sensitivity_analysis(full_series: ResidualSeries, n_obs_list: list[int], method: str = "ols_hac", alpha: float = 0.05) -> pd.DataFrame:
    """Análisis de sensibilidad por arco."""
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
                "n_obs": n, "arc_years": float(short.times_years[-1]),
                "dadt": res.dadt_au_my, "ci_lower": res.ci_lower,
                "ci_upper": res.ci_upper, "ci_width": res.ci_width,
                "std_err": res.std_error, "r2": res.r_squared,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)