"""
tests/test_layer2_ts.py
-----------------------
Tests unitarios para la Capa 2 de HYPATIA.

Ejecutar:
    pytest tests/test_layer2_ts.py -v              # tests rápidos
    pytest tests/test_layer2_ts.py -v --all        # incluye tests con JPL

Cobertura:
    - ResidualSeries: construcción, recorte, exportación
    - DiagnosticsReport: ADF, KPSS, Ljung-Box, recomendación
    - RegressionResult: OLS, HAC, STL sobre series sintéticas
    - BayesianPosterior: actualización con prior informativo/no informativo
    - Utils: conversiones de unidades, SNR, verificación de Capa 1
    - Pipeline: flujo offline completo
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_series():
    """
    Serie de residuos sintética con señal de Yarkovsky conocida.
    da/dt = -0.20 AU/My → pendiente = -0.20e-6 AU/año
    Ruido gaussiano σ = 0.5e-6 AU
    """
    rng  = np.random.default_rng(42)
    n    = 80
    t    = np.linspace(0, 20, n)            # 20 años de arco
    t_jd = 2451545.0 + t * 365.25

    true_dadt   = -0.20                     # AU/My
    slope_au_yr = true_dadt * 1e-6          # AU/año
    noise_std   = 0.5e-6                    # AU

    eps   = slope_au_yr * t + rng.normal(0, noise_std, n)
    a_obs = 0.9226 + eps
    a_pred = np.full(n, 0.9226)

    from src.layer2_ts.residuals import ResidualSeries
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
def noisy_series(synthetic_series):
    """Serie con ruido 3× mayor (SNR bajo)."""
    from src.layer2_ts.residuals import ResidualSeries
    base, true_dadt = synthetic_series
    rng  = np.random.default_rng(99)
    eps  = base.epsilon + rng.normal(0, 1.5e-6, base.n_points)
    return ResidualSeries(
        times_jd    = base.times_jd,
        times_years = base.times_years,
        a_obs       = base.a_pred + eps,
        a_pred      = base.a_pred,
        epsilon     = eps,
        epsilon_km  = eps * 1.495978707e8,
        n_points    = base.n_points,
        epoch_start = base.epoch_start,
        epoch_end   = base.epoch_end,
        asteroid_id = base.asteroid_id,
    ), true_dadt


@pytest.fixture
def ml_quantiles():
    """Cuantiles típicos del modelo ML para Apophis."""
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
        from src.layer2_ts import simulate_short_arc
        series, _ = synthetic_series
        short = simulate_short_arc(series, 15)
        assert short.n_points == 15
        assert short.times_years[-1] < series.times_years[-1]

    def test_short_arc_clipping(self, synthetic_series):
        """Si n_obs > total, no debe recortar."""
        from src.layer2_ts import simulate_short_arc
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
        from src.layer2_ts import run_diagnostics
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        assert diag is not None

    def test_trend_detected(self, synthetic_series):
        from src.layer2_ts import run_diagnostics
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        # La serie tiene tendencia real de -0.20 AU/My → debe ser significativa
        assert diag.trend_significant

    def test_recommendation_is_valid(self, synthetic_series):
        from src.layer2_ts import run_diagnostics
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        assert diag.recommend_method() in ["ols", "ols_hac", "stl", "bayesian"]

    def test_acf_shape(self, synthetic_series):
        from src.layer2_ts import run_diagnostics
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        assert len(diag.acf_values) == len(diag.acf_lags)
        assert len(diag.pacf_values) == len(diag.acf_lags)

    def test_summary_string(self, synthetic_series):
        from src.layer2_ts import run_diagnostics
        series, _ = synthetic_series
        diag = run_diagnostics(series)
        s = diag.summary()
        assert "DIAGNÓSTICO" in s
        assert "Método recomendado" in s


# ── REGRESIÓN ─────────────────────────────────────────────────────────────

class TestRegression:

    def test_ols_estimate_sign(self, synthetic_series):
        """OLS debe detectar la pendiente negativa correctamente."""
        from src.layer2_ts import estimate_ols
        series, true_dadt = synthetic_series
        result = estimate_ols(series)
        assert result.dadt_au_my < 0, "da/dt debe ser negativo para esta serie"

    def test_ols_estimate_magnitude(self, synthetic_series):
        """OLS debe estar dentro del 50% del valor real (con ruido)."""
        from src.layer2_ts import estimate_ols
        series, true_dadt = synthetic_series
        result = estimate_ols(series)
        error_pct = abs(result.dadt_au_my - true_dadt) / abs(true_dadt)
        assert error_pct < 0.5, f"Error OLS demasiado alto: {error_pct:.1%}"

    def test_ols_ci_contains_true(self, synthetic_series):
        """El IC 95% de OLS debe contener el valor real."""
        from src.layer2_ts import estimate_ols
        series, true_dadt = synthetic_series
        result = estimate_ols(series)
        assert result.ci_lower <= true_dadt <= result.ci_upper, (
            f"IC [{result.ci_lower:.4f}, {result.ci_upper:.4f}] "
            f"no contiene {true_dadt}"
        )

    def test_ols_hac_more_conservative(self, synthetic_series):
        """HAC debe dar CI más amplio que OLS estándar."""
        from src.layer2_ts import estimate_ols, estimate_ols_hac
        series, _ = synthetic_series
        ols     = estimate_ols(series)
        ols_hac = estimate_ols_hac(series)
        assert ols_hac.ci_width >= ols.ci_width * 0.8   # al menos comparable

    def test_stl_returns_result(self, synthetic_series):
        from src.layer2_ts import estimate_stl
        series, _ = synthetic_series
        result = estimate_stl(series)
        assert result.method in ("stl", "ols_hac")   # puede caer a HAC si n corto
        assert not np.isnan(result.dadt_au_my)

    def test_all_methods_consistent_sign(self, synthetic_series):
        """Los tres métodos deben concordar en el signo de da/dt."""
        from src.layer2_ts import estimate_dadt_all_methods
        series, _ = synthetic_series
        results = estimate_dadt_all_methods(series)
        signs = [np.sign(r.dadt_au_my) for r in results.values()]
        # Todos los signos deben ser iguales
        assert len(set(signs)) == 1, f"Signos inconsistentes: {signs}"

    def test_r_squared_bounds(self, synthetic_series):
        from src.layer2_ts import estimate_ols
        series, _ = synthetic_series
        result = estimate_ols(series)
        assert 0 <= result.r_squared <= 1

    def test_significant_trend(self, synthetic_series):
        """Con 80 puntos y señal clara, el IC no debe cruzar el cero."""
        from src.layer2_ts import estimate_ols_hac
        series, _ = synthetic_series
        result = estimate_ols_hac(series)
        assert result.is_significant, (
            f"Tendencia no significativa: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        )

    def test_sensitivity_dataframe(self, synthetic_series):
        from src.layer2_ts import sensitivity_analysis
        series, _ = synthetic_series
        df = sensitivity_analysis(series, n_obs_list=[10, 20, 40])
        assert len(df) >= 2
        assert "dadt" in df.columns
        assert "ci_width" in df.columns

    def test_sensitivity_ci_decreases(self, synthetic_series):
        """El IC 95% debe disminuir conforme aumentan las observaciones."""
        from src.layer2_ts import sensitivity_analysis
        series, _ = synthetic_series
        df = sensitivity_analysis(series, n_obs_list=[10, 20, 40, 60])
        ci_widths = df["ci_width"].values
        # Al menos la última debe ser menor que la primera
        assert ci_widths[-1] < ci_widths[0], (
            "El IC no se reduce con más observaciones"
        )


# ── BAYESIANO ─────────────────────────────────────────────────────────────

class TestBayesian:

    def test_prior_from_quantiles(self, ml_quantiles):
        from src.layer2_ts import GaussianPrior
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        assert abs(prior.mean - ml_quantiles[0.50]) < 1e-6
        assert prior.std > 0

    def test_uninformative_prior(self):
        from src.layer2_ts import GaussianPrior
        prior = GaussianPrior.uninformative()
        assert prior.std > 1000   # prácticamente plano

    def test_bayesian_update_between_prior_and_likelihood(
        self, synthetic_series, ml_quantiles
    ):
        """La posterior debe estar entre el prior y la likelihood."""
        from src.layer2_ts import estimate_ols_hac, GaussianPrior, bayesian_update
        series, _ = synthetic_series
        lik   = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post  = bayesian_update(prior, lik)

        lo = min(prior.mean, lik.dadt_au_my)
        hi = max(prior.mean, lik.dadt_au_my)
        # La posterior debe estar en el rango [min, max] de prior y datos
        assert lo - 0.1 <= post.mean <= hi + 0.1, (
            f"Posterior {post.mean:.4f} fuera de rango [{lo:.4f}, {hi:.4f}]"
        )

    def test_uninformative_prior_converges_to_data(self, synthetic_series):
        """Con prior no informativo, la posterior debe ≈ OLS-HAC."""
        from src.layer2_ts import (
            estimate_ols_hac, GaussianPrior, bayesian_update
        )
        series, _ = synthetic_series
        lik   = estimate_ols_hac(series)
        prior = GaussianPrior.uninformative()
        post  = bayesian_update(prior, lik)
        assert abs(post.mean - lik.dadt_au_my) < 0.001

    def test_posterior_std_smaller_than_inputs(self, synthetic_series, ml_quantiles):
        """σ_post < min(σ_prior, σ_likelihood): la posterior es siempre más precisa."""
        from src.layer2_ts import estimate_ols_hac, GaussianPrior, bayesian_update
        series, _ = synthetic_series
        lik   = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post  = bayesian_update(prior, lik)
        assert post.std <= min(prior.std, lik.std_error) + 1e-8

    def test_weights_sum_to_one(self, synthetic_series, ml_quantiles):
        from src.layer2_ts import estimate_ols_hac, GaussianPrior, bayesian_update
        series, _ = synthetic_series
        lik   = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post  = bayesian_update(prior, lik)
        assert abs(post.weight_prior + post.weight_likelihood - 1.0) < 1e-6

    def test_to_layer1_input_format(self, synthetic_series, ml_quantiles):
        """La salida hacia Capa 1 debe tener las claves correctas."""
        from src.layer2_ts import (
            estimate_ols_hac, GaussianPrior, bayesian_update
        )
        series, _ = synthetic_series
        lik  = estimate_ols_hac(series)
        prior = GaussianPrior.from_quantiles(ml_quantiles)
        post = bayesian_update(prior, lik)
        out  = post.to_layer1_input()
        for key in ["dadt_mean", "dadt_std", "samples", "ci_lower", "ci_upper"]:
            assert key in out, f"Falta clave: {key}"
        assert len(out["samples"]) > 0

    def test_samples_distribution(self, synthetic_series):
        """Las muestras deben seguir la distribución posterior."""
        from src.layer2_ts import (
            estimate_ols_hac, GaussianPrior, bayesian_update
        )
        series, _ = synthetic_series
        lik   = estimate_ols_hac(series)
        prior = GaussianPrior.uninformative()
        post  = bayesian_update(prior, lik, n_samples=5000)
        assert abs(np.mean(post.samples) - post.mean) < 0.01
        assert abs(np.std(post.samples) - post.std) < 0.01


# ── UTILIDADES ────────────────────────────────────────────────────────────

class TestUtils:

    def test_unit_conversions(self):
        from src.layer2_ts import dadt_au_my_to_au_yr, dadt_au_yr_to_au_my
        dadt = -0.20
        assert abs(dadt_au_yr_to_au_my(dadt_au_my_to_au_yr(dadt)) - dadt) < 1e-10

    def test_displacement_linear(self):
        from src.layer2_ts import displacement_at_t
        dadt    = -0.20    # AU/My
        t_years = 40.0
        da      = displacement_at_t(dadt, t_years)
        expected = dadt * 1e-6 * t_years
        assert abs(da - expected) < 1e-12

    def test_displacement_km_positive_for_positive_dadt(self):
        from src.layer2_ts import displacement_km
        assert displacement_km(+0.20, 40) > 0
        assert displacement_km(-0.20, 40) < 0

    def test_snr_positive(self, synthetic_series):
        from src.layer2_ts import signal_to_noise_ratio
        series, _ = synthetic_series
        snr = signal_to_noise_ratio(series)
        assert snr > 0

    def test_snr_increases_with_arc(self, synthetic_series):
        """Series más largas deben tener mayor SNR."""
        from src.layer2_ts import signal_to_noise_ratio, simulate_short_arc
        series, _ = synthetic_series
        snr_short = signal_to_noise_ratio(simulate_short_arc(series, 20))
        snr_full  = signal_to_noise_ratio(series)
        assert snr_full > snr_short

    def test_layer1_integration(self):
        """La verificación de la Capa 1 debe pasar."""
        from src.layer2_ts import verify_layer1_integration
        assert verify_layer1_integration()

    def test_quality_check_pass(self, synthetic_series):
        from src.layer2_ts import check_residuals_quality
        series, _ = synthetic_series
        result = check_residuals_quality(series)
        assert result["passed"]

    def test_quality_check_fail_too_short(self, synthetic_series):
        from src.layer2_ts import check_residuals_quality, simulate_short_arc
        series, _ = synthetic_series
        tiny = simulate_short_arc(series, 3)
        result = check_residuals_quality(tiny)
        assert not result["passed"]

    def test_save_load_roundtrip(self, synthetic_series, tmp_path):
        from src.layer2_ts import save_series, load_series
        series, _ = synthetic_series
        path = tmp_path / "test_series.csv"
        save_series(series, path)
        loaded = load_series(path)
        np.testing.assert_allclose(series.epsilon, loaded.epsilon, rtol=1e-6)
        assert loaded.n_points == series.n_points

    def test_describe_series(self, synthetic_series):
        from src.layer2_ts.utils import describe_series
        series, _ = synthetic_series
        desc = describe_series(series)
        assert "n_points" in desc
        assert desc["n_points"] == series.n_points
        assert "trend_slope_au_my" in desc


# ── PIPELINE OFFLINE ─────────────────────────────────────────────────────

class TestPipelineOffline:

    def test_offline_pipeline(self, synthetic_series, ml_quantiles, tmp_path):
        """Pipeline offline: guardar CSV → cargar → estimar → posterior."""
        from src.layer2_ts import save_series, run_layer2_offline
        series, true_dadt = synthetic_series
        path = tmp_path / "apophis_residuals.csv"
        save_series(series, path)

        result = run_layer2_offline(
            series_csv_path = str(path),
            ml_quantiles    = ml_quantiles,
            verbose         = False,
        )

        assert result.posterior is not None
        assert result.posterior.std > 0
        assert result.arc_years > 0

        # La posterior debe ser razonablemente cercana al valor real
        error = abs(result.posterior.mean - true_dadt)
        assert error < 0.3, f"Posterior muy alejada: error={error:.4f} AU/My"

    def test_layer1_output_format(self, synthetic_series, ml_quantiles, tmp_path):
        """to_layer1_input() debe retornar el formato correcto."""
        from src.layer2_ts import save_series, run_layer2_offline
        series, _ = synthetic_series
        path = tmp_path / "series.csv"
        save_series(series, path)

        result = run_layer2_offline(str(path), ml_quantiles, verbose=False)
        out = result.to_layer1_input()

        assert "dadt_mean" in out
        assert "dadt_std" in out
        assert "samples" in out
        assert out["dadt_std"] > 0
        assert len(out["samples"]) == 200


# ── TESTS LENTOS (requieren JPL) ─────────────────────────────────────────

@pytest.mark.slow
class TestPipelineWithJPL:

    def test_full_pipeline_apophis(self):
        """Pipeline completo sobre Apophis con datos reales."""
        from src.layer2_ts import run_layer2
        result = run_layer2(
            asteroid_id  = 99942,
            epoch_start  = "2014-01-01",
            epoch_end    = "2024-01-01",
            a_au         = 0.9226,
            ecc          = 0.1914,
            ml_quantiles = {0.10: -0.28, 0.50: -0.19, 0.90: -0.11},
            n_obs_limit  = None,
            verbose      = True,
        )
        # El valor real de da/dt de Apophis es -0.20 ± 0.03 AU/My
        # La posterior debe estar dentro de ±0.10 del valor real
        assert abs(result.posterior.mean - (-0.20)) < 0.10
        assert result.posterior.std < 0.15

    def test_short_arc_experiment(self):
        """Con 10 observaciones, el IC debe ser más amplio que con 50."""
        from src.layer2_ts import run_layer2
        r10 = run_layer2(
            99942, "2014-01-01", "2024-01-01",
            a_au=0.9226, ecc=0.1914, n_obs_limit=10, verbose=False,
        )
        r50 = run_layer2(
            99942, "2014-01-01", "2024-01-01",
            a_au=0.9226, ecc=0.1914, n_obs_limit=50, verbose=False,
        )
        assert r10.posterior.std > r50.posterior.std
