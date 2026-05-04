"""
layer2_ts
Capa 2: Series de tiempo y estimación de da/dt.
"""
# Pipeline
from .pipeline import run_layer2, run_layer2_offline, Layer2Result

# Residuos
from .residuals import build_residual_series, simulate_short_arc, ResidualSeries

# Diagnóstico
from .diagnostics import run_diagnostics, DiagnosticsReport

# Regresión
from .regression import (
    estimate_ols, estimate_ols_hac, estimate_stl,
    estimate_dadt_all_methods, sensitivity_analysis, RegressionResult,
)

# Bayesiano
from .bayesian import (
    GaussianPrior, BayesianPosterior, bayesian_update,
    full_bayesian_estimation, compare_posteriors_by_arc,
)

# Visualización
from .visualizer import (
    plot_residuals, plot_regression_comparison, plot_sensitivity,
    plot_bayesian_update, plot_stl_decomposition, plot_acf_pacf,
)

# Utilidades
from .utils import (
    verify_layer1_integration, check_residuals_quality,
    save_series, load_series, describe_series, signal_to_noise_ratio,
    displacement_at_t, displacement_km, dadt_au_my_to_au_yr, dadt_au_yr_to_au_my,
)

__all__ = [
    "run_layer2", "run_layer2_offline", "Layer2Result",
    "build_residual_series", "simulate_short_arc", "ResidualSeries",
    "run_diagnostics", "DiagnosticsReport",
    "estimate_ols", "estimate_ols_hac", "estimate_stl",
    "estimate_dadt_all_methods", "sensitivity_analysis", "RegressionResult",
    "GaussianPrior", "BayesianPosterior", "bayesian_update",
    "full_bayesian_estimation", "compare_posteriors_by_arc",
    "plot_residuals", "plot_regression_comparison", "plot_sensitivity",
    "plot_bayesian_update", "plot_stl_decomposition", "plot_acf_pacf",
    "verify_layer1_integration", "check_residuals_quality",
    "save_series", "load_series", "describe_series",
    "signal_to_noise_ratio", "displacement_at_t", "displacement_km",
    "dadt_au_my_to_au_yr", "dadt_au_yr_to_au_my",
]