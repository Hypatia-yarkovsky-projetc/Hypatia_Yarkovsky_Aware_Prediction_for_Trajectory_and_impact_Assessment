"""
test_layer2_ts.py
Tests unitarios y de integración para la Capa 2 de HYPATIA.

Estructura:
- Fixtures: Generación de datos sintéticos para pruebas aisladas.
- TestResidualSeries: Validación de estructura de residuos.
- TestDiagnostics: Validación de tests estadísticos.
- TestRegression: OLS, HAC, STL sobre datos sintéticos.
- TestBayesian: Lógica de actualización bayesiana.
- TestUtils: Conversión de unidades y verificación de Capa 1.
- TestPipeline: Flujo completo offline.

Ejecución:
  pytest tests/test_layer2_ts.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

# Imports de Layer 2 (respetando estructura src/)
from src.layer2_ts.residuals import ResidualSeries, simulate_short_arc
from src.layer2_ts.diagnostics import run_diagnostics
from src.layer2_ts.regression import (
    estimate_ols, estimate_ols_hac, estimate_stl, estimate_dadt_all_methods, sensitivity_analysis
)
from src.layer2_ts.bayesian import GaussianPrior, BayesianPosterior, bayesian_update
from src.layer2_ts.pipeline import run_layer2_offline
from src.layer2_ts.utils import (
    dadt_au_my_to_au_yr, dadt_au_yr_to_au_my,
    displacement_at_t, displacement_km,
    signal_to_noise_ratio, verify_layer1_integration,
    check_residuals_quality, save_series, load_series, describe_series
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_series():
    """Serie de residuos sintética con señal de Yarkovsky conocida."""
    rng = np.random.default_rng(42)
    n = 80
    t = np.linspace(0, 20, n)             # 20 años
    t_jd = 2451545.0 + t * 365.25

    true_dadt = -0.20                     # AU/My
    slope_au_yr = true_dadt * 1e-6        # AU/año
    noise_std = 0.5e-6                    # AU

    eps = slope_au_yr * t + rng.normal(0, noise_std, n)
    a_obs = 0.9226 + eps
    a_pred = np.full(n, 0.9226)

    return ResidualSeries(
        times_jd    = t_jd,
        times_years = t,
        a_obs       = a_obs,
        a_pred      = a_pred,
        epsilon     = eps,
        epsilon_km  = eps * 1.495978707e8,
        n_points    = n,
        epoch_start = "2004-01-01",
        epoch_end   = "2024-01-01",
        asteroid_id = 99942,
    ), true_dadt


@pytest.fixture
def ml_quantiles():
    """Cuantiles del modelo ML."""
    return {0.10: -0.28, 0.25: -0.24, 0.50: -0.19, 0.75: -0.15, 0.90: -0.11}


# ── RESIDUAL SERIES ───────────────────────────────────────────────────────

class TestResidualSeries:

    def test_dataframe_export(self, synthetic_series):
        series, _ = synthetic_series
        df = series.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == series.n_points
        assert "epsilon_au" in df.columns

    def test_summary_string(self, synthetic_series):
        series, _ = synthetic_series
        s = series.summary()
        assert "ResidualSeries" in s
        assert str(series.asteroid_id) in s

    def test_simulate_short_arc(self, synthetic_series):
        series, _ = synthetic_series
        short = simulate_short_arc(series, 15)
        assert short.n_points == 15
        assert short.times_years[-1] < series.times_years[-1]

    def test_short_arc_clipping(self, synthetic_series):
        """Si n_obs > total, no debe recortar."""
        series, _ = synthetic_series
        same = simulate_short_arc(series, 1000)
        assert same.n_points == series.n_points

    def test_epsilon_shape(self, synthetic_series):
        series, _ = synthetic_series
        assert series.epsilon.shape == (series.n_points,)
        assert series.epsilon_km.shape == (series.n_points,)


# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────

class TestDiagnostics:

    def test_runs_without_error(self, synthetic_series):
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        assert diag is not None

    def test_trend_detected(self, synthetic_series):
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        assert diag.trend_significant

    def test_recommendation_is_valid(self, synthetic_series):
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        assert diag.recommend_method() in ["ols", "ols_hac", "stl", "bayesian"]

    def test_acf_shape(self, synthetic_series):
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        assert len(diag.acf_values) == len(diag.acf_lags)
        assert len(diag.pacf_values) == len(diag.acf_lags)


# ── REGRESIÓN ─────────────────────────────────────────────────────────────

class TestRegression:

    def test_ols_estimate_sign(self, synthetic_series):
        """OLS debe detectar pendiente negativa."""
        series, _ = synthetic_series
        result = estimate_ols(series)
        assert result.dadt_au_my < 0

    def test_ols_estimate_magnitude(self, synthetic_series):
        """OLS dentro del 50% del valor real."""
        series, true_dadt = synthetic_series
        result = estimate_ols(series)
        error_pct = abs(result.dadt_au_my - true_dadt) / abs(true_dadt)
        assert error_pct < 0.5

    def test_ols_ci_contains_true(self, synthetic_series):
        """IC 95% debe contener valor real."""
        series, true_dadt = synthetic_series
        result = estimate_ols(series)
        assert result.ci_lower <= true_dadt <= result.ci_upper

    def test_ols_hac_more_conservative(self, synthetic_series):
        """HAC debe dar CI amplio."""
        series, _ = synthetic_series
        ols = estimate_ols(series)
        ols_hac = estimate_ols_hac(series)
        assert ols_hac.ci_width >= ols.ci_width * 0.8

    def test_stl_returns_result(self, synthetic_series):
        series, _ = synthetic_series
        result = estimate_stl(series)
        assert result.method in ("stl", "ols_hac")
        assert not np.isnan(result.dadt_au_my)

    def test_all_methods_consistent_sign(self, synthetic_series):
        """Concordancia de signo en métodos."""
        series, _ = synthetic_series
        results = estimate_dadt_all_methods(series)
        signs = [np.sign(r.dadt_au_my) for r in results.values()]
        assert len(set(signs)) == 1

    def test_sensitivity_dataframe(self, synthetic_series):
        series, _ = synthetic_series
        df = sensitivity_analysis(series, n_obs_list=[10, 20, 40])
        assert len(df) >= 2
        assert "dadt" in df.columns

    def test_sensitivity_ci_decreases(self, synthetic_series):
        """IC debe reducirse con más observaciones."""
        series, _ = synthetic_series
        df = sensitivity_analysis(series, n_obs_list=[10, 20, 40, 60])
        ci_widths = df["ci_width"].values
        assert ci_widths[-1] < ci_widths[0]


# ── BAYESIANO ─────────────────────────────────────────────────────────────

class TestBayesian:

    def test_prior_from_quantiles(self, ml_quantiles):
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        assert abs(prior.mean - ml_quantiles[0.50]) < 1e-6
        assert prior.std > 0

    def test_uninformative_prior(self):
        prior = GaussianPrior.uninformative()
        assert prior.std > 1000

    def test_bayesian_update_range(self, synthetic_series, ml_quantiles):
        """Posterior entre prior y likelihood."""
        series, _ = synthetic_series
        lik = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post = bayesian_update(prior, lik)

        lo = min(prior.mean, lik.dadt_au_my)
        hi = max(prior.mean, lik.dadt_au_my)
        assert lo - 0.1 <= post.mean <= hi + 0.1

    def test_uninformative_converges(self, synthetic_series):
        """Prior no informativo converge a datos."""
        series, _ = synthetic_series
        lik = estimate_ols_hac(series)
        prior = GaussianPrior.uninformative()
        post = bayesian_update(prior, lik)
        assert abs(post.mean - lik.dadt_au_my) < 0.001

    def test_posterior_precision(self, synthetic_series, ml_quantiles):
        """Posterior más precisa que inputs."""
        series, _ = synthetic_series
        lik = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post = bayesian_update(prior, lik)
        assert post.std <= min(prior.std, lik.std_error) + 1e-8

    def test_weights_sum_to_one(self, synthetic_series, ml_quantiles):
        series, _ = synthetic_series
        lik = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post = bayesian_update(prior, lik)
        assert abs(post.weight_prior + post.weight_likelihood - 1.0) < 1e-6

    def test_to_layer1_input(self, synthetic_series, ml_quantiles):
        """Formato correcto para Capa 1."""
        series, _ = synthetic_series
        lik = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post = bayesian_update(prior, lik)
        out = post.to_layer1_input()
        for key in ["dadt_mean", "dadt_std", "samples", "ci_lower", "ci_upper"]:
            assert key in out
        assert len(out["samples"]) > 0


# ── UTILIDADES ────────────────────────────────────────────────────────────

class TestUtils:

    def test_unit_conversions(self):
        dadt = -0.20
        assert abs(dadt_au_yr_to_au_my(dadt_au_my_to_au_yr(dadt)) - dadt) < 1e-10

    def test_displacement_linear(self):
        dadt = -0.20
        t_years = 40.0
        da = displacement_at_t(dadt, t_years)
        expected = dadt * 1e-6 * t_years
        assert abs(da - expected) < 1e-12

    def test_snr_positive(self, synthetic_series):
        series, _ = synthetic_series
        snr = signal_to_noise_ratio(series)
        assert snr > 0

    def test_layer1_integration(self):
        """Verificación de conexión con Capa 1."""
        assert verify_layer1_integration() is True

    def test_quality_check_pass(self, synthetic_series):
        series, _ = synthetic_series
        result = check_residuals_quality(series)
        assert result["passed"]

    def test_quality_check_fail(self, synthetic_series):
        """Serie corta debe fallar."""
        series, _ = synthetic_series
        tiny = simulate_short_arc(series, 3)
        result = check_residuals_quality(tiny)
        assert not result["passed"]

    def test_save_load_roundtrip(self, synthetic_series):
        series, _ = synthetic_series
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
            save_series(series, path)
            loaded = load_series(path)
            np.testing.assert_allclose(series.epsilon, loaded.epsilon, rtol=1e-6)
            assert loaded.n_points == series.n_points


# ── PIPELINE OFFLINE ─────────────────────────────────────────────────────

class TestPipelineOffline:

    def test_full_offline_flow(self, synthetic_series, ml_quantiles):
        """Guardar CSV -> Cargar -> Pipeline."""
        series, true_dadt = synthetic_series
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
            save_series(series, path)

            result = run_layer2_offline(
                series_csv_path = path,
                ml_quantiles    = ml_quantiles,
                verbose         = False,
            )

            assert result.posterior is not None
            assert result.posterior.std > 0
            error = abs(result.posterior.mean - true_dadt)
            assert error < 0.3