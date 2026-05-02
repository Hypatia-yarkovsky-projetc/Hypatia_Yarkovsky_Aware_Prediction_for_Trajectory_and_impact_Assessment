"""
pipeline.py
-----------
Pipeline orquestador de la Capa 2 de HYPATIA.

Ejecuta el flujo completo de estimación de da/dt en una sola llamada:

    1. Construir serie de residuos (Capa 1 + datos JPL)
    2. Diagnosticar la serie (estacionariedad, autocorrelación, etc.)
    3. Estimar da/dt con OLS, OLS-HAC y STL
    4. Actualización bayesiana con prior del modelo ML (Capa 3)
    5. Retornar la posterior para la Capa 1 (cono de incertidumbre)

Uso principal desde el pipeline maestro (src/pipeline.py):
    from src.layer2_ts.pipeline import run_layer2

    result = run_layer2(
        asteroid_id  = 99942,
        epoch_start  = '2004-06-19',
        epoch_end    = '2024-01-01',
        a_au         = 0.9226,
        ecc          = 0.1914,
        ml_quantiles = {0.10: -0.26, 0.50: -0.19, 0.90: -0.12},
        n_obs_limit  = 20,        # simular 20 observaciones iniciales
    )

    # La posterior va directamente a la Capa 1
    dadt_for_integrator = result['posterior'].to_layer1_input()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .residuals   import ResidualSeries, build_residual_series, simulate_short_arc
from .diagnostics import DiagnosticsReport, run_diagnostics
from .regression  import RegressionResult, estimate_dadt_all_methods, sensitivity_analysis
from .bayesian    import BayesianPosterior, GaussianPrior, full_bayesian_estimation
from .utils       import (
    verify_layer1_integration,
    check_residuals_quality,
    save_series,
    describe_series,
)
from ..layer1_ode.constants import DEFAULT_PERTURBERS


# ── Resultado completo de la Capa 2 ──────────────────────────────────────

@dataclass
class Layer2Result:
    """
    Contenedor del resultado completo del pipeline de la Capa 2.

    El campo más importante para la integración con Capa 1 es `posterior`,
    que contiene la distribución de da/dt lista para parametrizar el integrador.
    """
    # Serie de residuos
    series        : ResidualSeries

    # Diagnóstico estadístico
    diagnostics   : DiagnosticsReport

    # Estimaciones de los tres métodos
    regressions   : dict[str, RegressionResult]

    # Método recomendado por el diagnóstico
    best_method   : str

    # Posterior bayesiana (integra prior ML + datos)
    posterior     : BayesianPosterior

    # Análisis de sensibilidad (si se corrió)
    sensitivity_df: Optional[object] = None

    # Metadatos
    asteroid_id   : int = 0
    n_obs_used    : Optional[int] = None
    arc_years     : float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  RESULTADO CAPA 2 — HYPATIA",
            "=" * 60,
            self.series.summary(),
            "",
            self.diagnostics.summary(),
            "",
            f"  Estimaciones de da/dt:",
            f"  {'─'*50}",
        ]
        for name, r in self.regressions.items():
            marker = " ← recomendado" if name == self.best_method else ""
            lines.append(f"  {r.summary()}{marker}")
            lines.append("")
        lines.append(self.posterior.summary())
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_layer1_input(self) -> dict:
        """
        Interfaz de salida hacia la Capa 1.
        Empaqueta la posterior en el formato que espera generate_uncertainty_cone().

        Returns:
            dict {
                'dadt_mean': float,   # da/dt media posterior [AU/My]
                'dadt_std' : float,   # σ posterior [AU/My]
                'samples'  : ndarray, # muestras MC para el cono
                'ci_lower' : float,
                'ci_upper' : float,
            }
        """
        return {
            "dadt_mean": self.posterior.mean,
            "dadt_std" : self.posterior.std,
            "samples"  : self.posterior.sample_dadt(200),
            "ci_lower" : self.posterior.ci_lower,
            "ci_upper" : self.posterior.ci_upper,
        }


# ── Función principal del pipeline ───────────────────────────────────────

def run_layer2(
    asteroid_id    : int | str,
    epoch_start    : str,
    epoch_end      : str,
    a_au           : float,
    ecc            : float,
    ml_quantiles   : Optional[dict] = None,
    n_obs_limit    : Optional[int] = None,
    perturbers     : list[str] = DEFAULT_PERTURBERS,
    obs_step       : str = "30d",
    run_sensitivity: bool = False,
    n_obs_sensitivity: list[int] = [5, 10, 20, 30, 50],
    save_series_path: Optional[str] = None,
    verbose        : bool = True,
) -> Layer2Result:
    """
    Ejecuta el pipeline completo de la Capa 2.

    Args:
        asteroid_id    : ID JPL del asteroide (ej. 99942)
        epoch_start    : inicio del arco histórico 'YYYY-MM-DD'
        epoch_end      : fin del arco 'YYYY-MM-DD'
        a_au           : semieje mayor nominal [AU]
        ecc            : excentricidad nominal
        ml_quantiles   : cuantiles del modelo ML de la Capa 3
                         {0.10: val, 0.25: val, 0.50: val, 0.75: val, 0.90: val}
                         Si None → prior no informativo (solo datos)
        n_obs_limit    : limitar a las primeras N observaciones
                         (simula objeto recién descubierto)
        perturbers     : cuerpos perturbadores para el integrador
        obs_step       : paso entre observaciones JPL
        run_sensitivity: correr análisis de sensibilidad (más lento)
        n_obs_sensitivity: lista de arcos para el análisis de sensibilidad
        save_series_path: guardar la serie de residuos a CSV si se especifica
        verbose        : imprimir progreso detallado

    Returns:
        Layer2Result con todos los resultados y la posterior lista para Capa 1
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  HYPATIA — CAPA 2: SERIES DE TIEMPO")
        print("=" * 60)

    # ── Paso 0: verificar integración con Capa 1 ─────────────────────────
    if not verify_layer1_integration():
        raise RuntimeError("Integración con Capa 1 fallida. Verifica la instalación.")

    # ── Paso 1: construir serie de residuos ───────────────────────────────
    if verbose:
        print("\n[Paso 1/5] Construyendo serie de residuos orbitales...")

    full_series = build_residual_series(
        asteroid_id  = asteroid_id,
        epoch_start  = epoch_start,
        epoch_end    = epoch_end,
        a_au         = a_au,
        ecc          = ecc,
        perturbers   = perturbers,
        obs_step     = obs_step,
        verbose      = verbose,
    )

    # Guardar si se solicitó
    if save_series_path:
        save_series(full_series, save_series_path)

    # Aplicar límite de observaciones
    series = (
        simulate_short_arc(full_series, n_obs_limit)
        if n_obs_limit and n_obs_limit < full_series.n_points
        else full_series
    )

    # ── Paso 2: verificar calidad de la serie ─────────────────────────────
    quality = check_residuals_quality(series)
    if not quality["passed"]:
        raise ValueError(
            f"Serie de residuos no válida para análisis: {quality['errors']}"
        )

    # ── Paso 3: diagnóstico estadístico ───────────────────────────────────
    if verbose:
        print("\n[Paso 2/5] Ejecutando diagnóstico estadístico...")

    diag = run_diagnostics(series)

    if verbose:
        print(diag.summary())

    best_method = diag.recommend_method()

    # ── Paso 4: estimación con los tres métodos ────────────────────────────
    if verbose:
        print("\n[Paso 3/5] Estimando da/dt con OLS, OLS-HAC y STL...")

    regressions = estimate_dadt_all_methods(series)

    # ── Paso 5: actualización bayesiana ───────────────────────────────────
    if verbose:
        print("\n[Paso 4/5] Actualización bayesiana con prior ML...")

    best_reg = regressions.get(best_method, regressions["ols_hac"])
    posterior = full_bayesian_estimation(
        regression_result = best_reg,
        ml_quantiles      = ml_quantiles,
        verbose           = verbose,
    )

    # ── Paso 6: análisis de sensibilidad (opcional) ────────────────────────
    sensitivity_df = None
    if run_sensitivity and full_series.n_points > max(n_obs_sensitivity, default=0):
        if verbose:
            print("\n[Paso 5/5] Análisis de sensibilidad por arco...")
        sensitivity_df = sensitivity_analysis(
            full_series,
            n_obs_list = [n for n in n_obs_sensitivity
                          if n <= full_series.n_points],
            method     = best_method,
        )

    # ── Resultado final ───────────────────────────────────────────────────
    result = Layer2Result(
        series         = series,
        diagnostics    = diag,
        regressions    = regressions,
        best_method    = best_method,
        posterior      = posterior,
        sensitivity_df = sensitivity_df,
        asteroid_id    = int(asteroid_id),
        n_obs_used     = series.n_points,
        arc_years      = float(series.times_years[-1]),
    )

    if verbose:
        print("\n" + result.summary())

    return result


# ── Modo offline: correr sobre CSV ya guardado ────────────────────────────

def run_layer2_offline(
    series_csv_path: str,
    ml_quantiles   : Optional[dict] = None,
    n_obs_limit    : Optional[int] = None,
    verbose        : bool = True,
) -> Layer2Result:
    """
    Versión offline del pipeline: carga la serie de residuos desde CSV
    en lugar de descargarlo desde JPL Horizons.

    Útil para reproducir resultados sin conexión a internet,
    o cuando el Integrante 2 trabaja con los datos ya descargados
    por el Integrante 1 durante la sesión de setup.

    Args:
        series_csv_path : ruta al CSV guardado con save_series()
        ml_quantiles    : cuantiles del modelo ML
        n_obs_limit     : limitar a N observaciones
        verbose         : imprimir progreso

    Returns:
        Layer2Result
    """
    from .utils import load_series

    if verbose:
        print(f"\n[HYPATIA L2] Modo offline — cargando: {series_csv_path}")

    full_series = load_series(series_csv_path)

    series = (
        simulate_short_arc(full_series, n_obs_limit)
        if n_obs_limit and n_obs_limit < full_series.n_points
        else full_series
    )

    quality = check_residuals_quality(series)
    if not quality["passed"]:
        raise ValueError(f"Serie no válida: {quality['errors']}")

    diag        = run_diagnostics(series)
    best_method = diag.recommend_method()
    regressions = estimate_dadt_all_methods(series)
    best_reg    = regressions.get(best_method, regressions["ols_hac"])
    posterior   = full_bayesian_estimation(best_reg, ml_quantiles, verbose=verbose)

    return Layer2Result(
        series         = series,
        diagnostics    = diag,
        regressions    = regressions,
        best_method    = best_method,
        posterior      = posterior,
        asteroid_id    = series.asteroid_id,
        n_obs_used     = series.n_points,
        arc_years      = float(series.times_years[-1]),
    )
