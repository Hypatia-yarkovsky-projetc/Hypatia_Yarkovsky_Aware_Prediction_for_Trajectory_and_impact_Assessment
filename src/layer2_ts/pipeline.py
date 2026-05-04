"""
pipeline.py
Orquestador de la Capa 2.
Flujo: Residuos -> Diagnóstico -> Regresión -> Bayesiano -> Salida Capa 1.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from .residuals import ResidualSeries, build_residual_series, simulate_short_arc
from .diagnostics import DiagnosticsReport, run_diagnostics
from .regression import RegressionResult, estimate_dadt_all_methods, sensitivity_analysis
from .bayesian import BayesianPosterior, GaussianPrior, full_bayesian_estimation
from .utils import verify_layer1_integration, check_residuals_quality, save_series
from ..layer1_ode.constants import DEFAULT_PERTURBERS

@dataclass
class Layer2Result:
    """Contenedor de resultados de la Capa 2."""
    series: ResidualSeries
    diagnostics: DiagnosticsReport
    regressions: dict[str, RegressionResult]
    best_method: str
    posterior: BayesianPosterior
    sensitivity_df: Optional[object] = None
    asteroid_id: int = 0; n_obs_used: Optional[int] = None; arc_years: float = 0.0

    def summary(self) -> str:
        return (
            f"RESULTADO CAPA 2 | {self.series.summary()}\n"
            f"Diagnóstico: {self.diagnostics.summary()}\n"
            f"Método: {self.best_method.upper()} | {self.posterior.summary()}"
        )

    def to_layer1_input(self) -> dict:
        return {
            "dadt_mean": self.posterior.mean, "dadt_std": self.posterior.std,
            "samples": self.posterior.sample_dadt(200),
            "ci_lower": self.posterior.ci_lower, "ci_upper": self.posterior.ci_upper,
        }

def run_layer2(
    asteroid_id: int | str, epoch_start: str, epoch_end: str,
    a_au: float, ecc: float, ml_quantiles: Optional[dict] = None,
    n_obs_limit: Optional[int] = None, perturbers: list[str] = DEFAULT_PERTURBERS,
    obs_step: str = "30d", run_sensitivity: bool = False,
    n_obs_sensitivity: list[int] = [5, 10, 20, 30], save_series_path: Optional[str] = None,
    verbose: bool = True,
) -> Layer2Result:
    """Pipeline completo de la Capa 2."""
    if verbose: print("\nHYPATIA — CAPA 2: SERIES DE TIEMPO")

    if not verify_layer1_integration():
        raise RuntimeError("Integración con Capa 1 fallida.")

    if verbose: print("[Paso 1] Construyendo serie de residuos...")
    full_series = build_residual_series(
        asteroid_id, epoch_start, epoch_end, a_au, ecc,
        perturbers, obs_step, verbose=verbose,
    )
    if save_series_path: save_series(full_series, save_series_path)

    series = simulate_short_arc(full_series, n_obs_limit) if n_obs_limit else full_series

    quality = check_residuals_quality(series)
    if not quality["passed"]: raise ValueError(f"Serie inválida: {quality['errors']}")

    if verbose: print("[Paso 2] Diagnóstico...")
    diag = run_diagnostics(series)
    best_method = diag.recommend_method()

    if verbose: print(f"[Paso 3] Regresión ({best_method})...")
    regressions = estimate_dadt_all_methods(series)
    best_reg = regressions.get(best_method, regressions["ols_hac"])

    if verbose: print("[Paso 4] Bayesiano...")
    posterior = full_bayesian_estimation(best_reg, ml_quantiles, verbose=verbose)

    sensitivity_df = None
    if run_sensitivity and full_series.n_points > max(n_obs_sensitivity, default=0):
        if verbose: print("[Paso 5] Sensibilidad...")
        sensitivity_df = sensitivity_analysis(full_series, n_obs_sensitivity, best_method)

    return Layer2Result(
        series=series, diagnostics=diag, regressions=regressions,
        best_method=best_method, posterior=posterior, sensitivity_df=sensitivity_df,
        asteroid_id=int(asteroid_id), n_obs_used=series.n_points, arc_years=float(series.times_years[-1]),
    )

def run_layer2_offline(series_csv_path: str, ml_quantiles: Optional[dict] = None, n_obs_limit: Optional[int] = None, verbose: bool = True) -> Layer2Result:
    """Versión offline desde CSV."""
    from .utils import load_series
    if verbose: print(f"Modo offline: {series_csv_path}")
    
    full_series = load_series(series_csv_path)
    series = simulate_short_arc(full_series, n_obs_limit) if n_obs_limit else full_series
    
    diag = run_diagnostics(series)
    best_method = diag.recommend_method()
    regressions = estimate_dadt_all_methods(series)
    best_reg = regressions.get(best_method, regressions["ols_hac"])
    posterior = full_bayesian_estimation(best_reg, ml_quantiles, verbose=False)
    
    return Layer2Result(
        series=series, diagnostics=diag, regressions=regressions,
        best_method=best_method, posterior=posterior,
        asteroid_id=series.asteroid_id, n_obs_used=series.n_points, arc_years=float(series.times_years[-1]),
    )