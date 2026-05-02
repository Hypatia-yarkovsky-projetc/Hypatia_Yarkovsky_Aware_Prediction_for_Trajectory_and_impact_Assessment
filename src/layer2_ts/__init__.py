"""
layer2_ts
---------
Capa 2 de HYPATIA: Series de tiempo y estimación de da/dt.

Detecta y cuantifica la huella temporal del efecto Yarkovsky en los
residuos orbitales históricos, produciendo una distribución posterior
de da/dt que alimenta el integrador de la Capa 1.

API pública:

    # Flujo completo (recomendado)
    from src.layer2_ts import run_layer2
    result = run_layer2(asteroid_id=99942, ...)
    dadt   = result.to_layer1_input()

    # Módulos individuales
    from src.layer2_ts import (
        build_residual_series,   # Capa 2 ← Capa 1: calcular residuos
        run_diagnostics,         # Diagnóstico estadístico
        estimate_dadt_all_methods, # OLS, OLS-HAC, STL
        full_bayesian_estimation,  # Prior ML + likelihood → posterior
        plot_residuals,          # Visualización principal
    )

Integración con las otras capas:
    ← Capa 1: propagate_from_state(), fetch_ephemeris_arc(), semi_major_axis()
    → Capa 1: posterior.to_layer1_input() → dadt_mean, dadt_std, samples
    ← Capa 3: ml_quantiles → GaussianPrior para bayesian_update()
"""

# Pipeline principal
from .pipeline import run_layer2, run_layer2_offline, Layer2Result

# Residuos
from .residuals import (
    build_residual_series,
    simulate_short_arc,
    ResidualSeries,
)

# Diagnóstico
from .diagnostics import run_diagnostics, DiagnosticsReport

# Regresión
from .regression import (
    estimate_ols,
    estimate_ols_hac,
    estimate_stl,
    estimate_dadt_all_methods,
    sensitivity_analysis,
    RegressionResult,
)

# Bayesiano
from .bayesian import (
    GaussianPrior,
    BayesianPosterior,
    bayesian_update,
    full_bayesian_estimation,
    compare_posteriors_by_arc,
)

# Visualización
from .visualizer import (
    plot_residuals,
    plot_regression_comparison,
    plot_sensitivity,
    plot_bayesian_update,
    plot_stl_decomposition,
    plot_acf_pacf,
)

# Utilidades
from .utils import (
    verify_layer1_integration,
    check_residuals_quality,
    save_series,
    load_series,
    describe_series,
    signal_to_noise_ratio,
    displacement_at_t,
    displacement_km,
    dadt_au_my_to_au_yr,
    dadt_au_yr_to_au_my,
)

__all__ = [
    # Pipeline
    "run_layer2", "run_layer2_offline", "Layer2Result",
    # Residuos
    "build_residual_series", "simulate_short_arc", "ResidualSeries",
    # Diagnóstico
    "run_diagnostics", "DiagnosticsReport",
    # Regresión
    "estimate_ols", "estimate_ols_hac", "estimate_stl",
    "estimate_dadt_all_methods", "sensitivity_analysis", "RegressionResult",
    # Bayesiano
    "GaussianPrior", "BayesianPosterior", "bayesian_update",
    "full_bayesian_estimation", "compare_posteriors_by_arc",
    # Visualización
    "plot_residuals", "plot_regression_comparison", "plot_sensitivity",
    "plot_bayesian_update", "plot_stl_decomposition", "plot_acf_pacf",
    # Utilidades
    "verify_layer1_integration", "check_residuals_quality",
    "save_series", "load_series", "describe_series",
    "signal_to_noise_ratio", "displacement_at_t", "displacement_km",
    "dadt_au_my_to_au_yr", "dadt_au_yr_to_au_my",
]
