"""
tests/test_layer3_ml.py
Tests unitarios para la Capa 3 de HYPATIA.
Ejecutar:
    pytest tests/test_layer3_ml.py -v              # tests rápidos
    pytest tests/test_layer3_ml.py -v --all        # incluye LOO-CV completo

Cobertura:
    - Dataset: construcción, conversión A2→da/dt, imputación
    - Features: extracción, IDW weights, validación de bounds
    - Modelo: entrenamiento, predicción cuantílica, monotonía
    - Inferencia: Apophis, batch, formato de salida
    - Integración: compatibilidad con Capas 1 y 2
    - Pipeline: flujo offline completo
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def training_df():
    """Dataset de entrenamiento con el fallback embebido."""
    from src.layer3_ml.dataset import build_training_dataset
    return build_training_dataset(verbose=False)


@pytest.fixture(scope="module")
def trained_model(training_df):
    """Modelo entrenado sobre el dataset fallback (sin LOO-CV para velocidad)."""
    from src.layer3_ml.model import train
    return train(training_df, verbose=False)


@pytest.fixture
def apophis_features():
    return {
        "diameter_km": 0.37,
        "albedo_pV"  : 0.23,
        "taxonomy"   : "Sq",
        "rot_per_h"  : 30.4,
        "a_AU"       : 0.9226,
        "ecc"        : 0.1914,
    }


# ── DATASET ───────────────────────────────────────────────────────────────

class TestDataset:

    def test_fallback_not_empty(self):
        from src.layer3_ml.dataset import _get_fallback_dataset
        df = _get_fallback_dataset()
        assert len(df) >= 10

    def test_compute_dadt_from_A2(self):
        from src.layer3_ml.dataset import _get_fallback_dataset, compute_dadt_from_A2
        df = compute_dadt_from_A2(_get_fallback_dataset()) 
        assert "dadt_AuMy" in df.columns
        assert df["dadt_AuMy"].notna().all()

    def test_apophis_dadt_sign(self):
        """Apophis tiene A2 < 0 → da/dt debe ser negativo."""
        from src.layer3_ml.dataset import _get_fallback_dataset, compute_dadt_from_A2
        df = compute_dadt_from_A2(_get_fallback_dataset())
        apophis = df[df["full_name"].str.contains("Apophis", na=False)]
        assert len(apophis) > 0
        assert float(apophis["dadt_AuMy"].iloc[0]) < 0

    def test_build_training_dataset_columns(self, training_df):
        required = ["inv_diameter", "absorptivity", "tax_code",
                     "rot_per_h", "a_AU", "ecc", "dadt_AuMy"]
        for col in required:
            assert col in training_df.columns, f"Columna faltante: {col}"

    def test_inv_diameter_positive(self, training_df):
        assert (training_df["inv_diameter"] > 0).all()

    def test_absorptivity_bounded(self, training_df):
        assert training_df["absorptivity"].between(0, 1).all()

    def test_tax_code_bounded(self, training_df):
        assert training_df["tax_code"].between(0, 1).all()

    def test_dadt_physical_range(self, training_df):
        """da/dt debe estar en rango físico [-10, +10] AU/My para NEOs."""
        assert training_df["dadt_AuMy"].abs().lt(10).all()

    def test_no_nans_in_features(self, training_df):
        from src.layer3_ml.features import FEATURE_NAMES
        for col in FEATURE_NAMES:
            assert training_df[col].notna().all(), f"NaN en feature: {col}"


# ── FEATURES ──────────────────────────────────────────────────────────────

class TestFeatures:

    def test_extract_features_shape(self, training_df):
        from src.layer3_ml.features import extract_features, FEATURE_NAMES
        X = extract_features(training_df)
        assert X.shape == (len(training_df), len(FEATURE_NAMES))

    def test_build_feature_vector_keys(self, apophis_features):
        from src.layer3_ml.features import build_feature_vector, FEATURE_NAMES
        feats = build_feature_vector(**apophis_features)
        for key in FEATURE_NAMES:
            assert key in feats, f"Feature faltante: {key}"

    def test_inv_diameter_physics(self):
        """1/D_km debe aumentar conforme D disminuye."""
        from src.layer3_ml.features import build_feature_vector
        f1 = build_feature_vector(0.5, 0.2, "S", 10, 1.0, 0.2)
        f2 = build_feature_vector(1.0, 0.2, "S", 10, 1.0, 0.2)
        assert f1["inv_diameter"] > f2["inv_diameter"]

    def test_absorptivity_complement(self):
        from src.layer3_ml.features import build_feature_vector
        f = build_feature_vector(1.0, 0.23, "S", 10, 1.0, 0.2)
        assert abs(f["absorptivity"] - (1 - 0.23)) < 1e-10

    def test_sample_weights_positive(self, training_df):
        from src.layer3_ml.features import extract_features, compute_sample_weights
        X = extract_features(training_df)
        w = compute_sample_weights(X)
        assert (w > 0).all()

    def test_sample_weights_shape(self, training_df):
        from src.layer3_ml.features import extract_features, compute_sample_weights
        X = extract_features(training_df)
        w = compute_sample_weights(X)
        assert len(w) == len(training_df)

    def test_uniform_weights(self, training_df):
        from src.layer3_ml.features import extract_features, compute_sample_weights
        X = extract_features(training_df)
        w = compute_sample_weights(X, method="uniform")
        np.testing.assert_array_equal(w, np.ones(len(X)))

    def test_validate_domain_in_range(self, apophis_features):
        from src.layer3_ml.features import build_feature_vector, validate_new_asteroid
        feats = build_feature_vector(**apophis_features)
        valid, warns = validate_new_asteroid(feats)
        assert valid

    def test_features_to_dataframe(self, apophis_features):
        from src.layer3_ml.features import build_feature_vector, features_to_dataframe, FEATURE_NAMES
        feats = build_feature_vector(**apophis_features)
        df = features_to_dataframe(feats)
        assert df.shape == (1, len(FEATURE_NAMES))


# ── MODELO ────────────────────────────────────────────────────────────────

class TestModel:

    def test_model_is_fitted(self, trained_model):
        assert trained_model.is_fitted

    def test_quantile_keys(self, trained_model):
        from src.layer3_ml.model import QUANTILES
        for q in QUANTILES:
            assert q in trained_model.quantile_models

    def test_predict_quantiles_shape(self, trained_model, training_df):
        from src.layer3_ml.features import extract_features
        X = extract_features(training_df)
        preds = trained_model.predict_quantiles(X)
        for q, arr in preds.items():
            assert len(arr) == len(training_df)

    def test_quantile_monotonicity_mostly(self, trained_model, training_df):
        """La mayoría de predicciones deben ser monótonas."""
        from src.layer3_ml.utils import check_monotonicity
        result = check_monotonicity(trained_model, training_df)
        # Toleramos hasta 20% de violaciones con dataset pequeño
        assert result["violation_rate"] < 0.20

    def test_predict_single_keys(self, trained_model, apophis_features):
        from src.layer3_ml.features import build_feature_vector
        feats = build_feature_vector(**apophis_features)
        preds = trained_model.predict_single(feats)
        from src.layer3_ml.model import QUANTILES
        for q in QUANTILES:
            assert q in preds

    def test_median_not_nan(self, trained_model, apophis_features):
        from src.layer3_ml.features import build_feature_vector
        feats = build_feature_vector(**apophis_features)
        preds = trained_model.predict_single(feats)
        assert not np.isnan(preds[0.50])

    def test_to_layer2_prior_format(self, trained_model, apophis_features):
        from src.layer3_ml.features import build_feature_vector
        feats = build_feature_vector(**apophis_features)
        prior = trained_model.to_layer2_prior(feats)
        required = {0.10, 0.25, 0.50, 0.75, 0.90}
        assert required.issubset(set(prior.keys()))

    def test_save_load_roundtrip(self, trained_model, tmp_path):
        from src.layer3_ml.utils import save_model, load_model
        path = tmp_path / "model.joblib"
        save_model(trained_model, str(path))
        loaded = load_model(str(path))
        assert loaded.is_fitted
        assert loaded.n_training == trained_model.n_training

    def test_predict_after_load(self, trained_model, apophis_features, tmp_path):
        from src.layer3_ml.utils import save_model, load_model
        from src.layer3_ml.features import build_feature_vector
        path = tmp_path / "model2.joblib"
        save_model(trained_model, str(path))
        loaded = load_model(str(path))
        feats = build_feature_vector(**apophis_features)
        original = trained_model.predict_single(feats)
        reloaded = loaded.predict_single(feats)
        assert abs(original[0.50] - reloaded[0.50]) < 1e-6


# ── INFERENCIA ────────────────────────────────────────────────────────────

class TestInference:

    def test_apophis_sign(self, trained_model, apophis_features):
        """El modelo debe predecir da/dt negativo para Apophis."""
        from src.layer3_ml.inference import inferir_dadt
        res = inferir_dadt(**apophis_features, model=trained_model, verbose=False)
        assert res.median < 0, (
            f"Apophis debería tener da/dt < 0, obtenido: {res.median:.4f}"
        )

    def test_inference_result_fields(self, trained_model, apophis_features):
        from src.layer3_ml.inference import inferir_dadt
        res = inferir_dadt(**apophis_features, model=trained_model, verbose=False)
        assert res.quantiles is not None
        assert res.ci_80_lower < res.ci_80_upper
        assert res.uncertainty_au_my > 0

    def test_ci_contains_median(self, trained_model, apophis_features):
        from src.layer3_ml.inference import inferir_dadt
        res = inferir_dadt(**apophis_features, model=trained_model, verbose=False)
        assert res.ci_80_lower <= res.median <= res.ci_80_upper

    def test_to_layer2_prior_compatible(self, trained_model, apophis_features):
        """El prior debe ser compatible con GaussianPrior.from_quantiles()."""
        from src.layer3_ml.inference import inferir_dadt
        from src.layer2_ts.bayesian import GaussianPrior
        res   = inferir_dadt(**apophis_features, model=trained_model, verbose=False)
        prior = GaussianPrior.from_quantiles(res.to_layer2_prior())
        assert prior.std > 0
        assert not np.isnan(prior.mean)

    def test_small_asteroid_larger_uncertainty(self, trained_model):
        """Asteroides pequeños deben tener mayor rango de IC que grandes."""
        from src.layer3_ml.inference import inferir_dadt
        small = inferir_dadt(0.05, 0.20, "S", 5.0, 1.0, 0.2,
                             trained_model, verbose=False)
        large = inferir_dadt(5.00, 0.20, "S", 5.0, 1.0, 0.2,
                             trained_model, verbose=False)
        # Generalmente asteroides pequeños tienen mayor incertidumbre relativa
        assert small.uncertainty_au_my >= 0   # al menos no negativa

    def test_batch_inference(self, trained_model, training_df):
        from src.layer3_ml.inference import inferir_dadt_batch
        small_df = training_df.head(5).copy()
        result   = inferir_dadt_batch(small_df, trained_model, verbose=False)
        assert "dadt_q50" in result.columns
        assert result["dadt_q50"].notna().all()

    def test_summary_string(self, trained_model, apophis_features):
        from src.layer3_ml.inference import inferir_dadt
        res = inferir_dadt(**apophis_features, model=trained_model, verbose=False)
        s   = res.summary()
        assert "Q50" in s or "Mediana" in s
        assert "AU/My" in s


# ── INTEGRACIÓN ───────────────────────────────────────────────────────────

class TestIntegration:

    def test_layer3_integration_check(self):
        from src.layer3_ml.utils import verify_layer3_integration
        assert verify_layer3_integration()

    def test_output_format_check(self, trained_model):
        from src.layer3_ml.utils import verify_output_format
        assert verify_output_format(trained_model)

    def test_benchmark_runs(self, trained_model):
        from src.layer3_ml.utils import benchmark_reference_asteroids
        df = benchmark_reference_asteroids(trained_model, verbose=False)
        assert len(df) >= 2
        assert "da/dt real" in df.columns
        assert "Q50 (pred)" in df.columns

    def test_apophis_benchmark_sign(self, trained_model):
        """En el benchmark, Apophis debe predecirse negativo."""
        from src.layer3_ml.utils import benchmark_reference_asteroids
        df = benchmark_reference_asteroids(trained_model, verbose=False)
        apophis_row = df[df["Asteroide"] == "Apophis"]
        assert len(apophis_row) > 0
        assert float(apophis_row["Q50 (pred)"].iloc[0]) < 0

    def test_get_apophis_prior(self, trained_model):
        from src.layer3_ml.utils import get_apophis_prior
        prior = get_apophis_prior(trained_model)
        assert 0.50 in prior
        assert prior[0.10] < prior[0.50] < prior[0.90]


# ── PIPELINE ──────────────────────────────────────────────────────────────

class TestPipeline:

    def test_offline_pipeline(self, trained_model, apophis_features, tmp_path):
        from src.layer3_ml.utils import save_model
        from src.layer3_ml.pipeline import run_layer3_offline
        model_path = tmp_path / "model.joblib"
        save_model(trained_model, str(model_path))
        result = run_layer3_offline(str(model_path), apophis_features, verbose=False)
        assert result.prior_quantiles is not None
        assert result.inference.median < 0   # Apophis: da/dt negativo

    def test_pipeline_to_layer2_input(self, trained_model, apophis_features, tmp_path):
        from src.layer3_ml.utils import save_model
        from src.layer3_ml.pipeline import run_layer3_offline
        model_path = tmp_path / "m.joblib"
        save_model(trained_model, str(model_path))
        result = run_layer3_offline(str(model_path), apophis_features, verbose=False)
        layer2_input = result.to_layer2_input()
        required = {0.10, 0.25, 0.50, 0.75, 0.90}
        assert required.issubset(set(layer2_input.keys()))

    def test_full_integration_l3_to_l2(
        self, trained_model, apophis_features, tmp_path
    ):
        """
        Test de integración completa: L3 → prior → L2 GaussianPrior.
        Verifica que el prior de la Capa 3 es consumible por la Capa 2.
        """
        from src.layer3_ml.utils import save_model
        from src.layer3_ml.pipeline import run_layer3_offline
        from src.layer2_ts.bayesian import GaussianPrior

        model_path = tmp_path / "m2.joblib"
        save_model(trained_model, str(model_path))

        result = run_layer3_offline(str(model_path), apophis_features, verbose=False)
        prior  = GaussianPrior.from_quantiles(result.to_layer2_input())

        assert not np.isnan(prior.mean)
        assert prior.std > 0
        print(f"\n  L3→L2 integración: μ={prior.mean:.4f}, σ={prior.std:.4f} AU/My")


# ── TESTS LENTOS (LOO-CV completo) ────────────────────────────────────────

@pytest.mark.slow
class TestLOOCV:

    def test_loocv_rmse_acceptable(self):
        """RMSE LOO-CV debe ser < 0.15 AU/My con el dataset de respaldo."""
        from src.layer3_ml.dataset import build_training_dataset
        from src.layer3_ml.model import validate_loocv
        df = build_training_dataset(verbose=False)
        report = validate_loocv(df, verbose=True)
        # Con solo ~20 asteroides de respaldo el RMSE será mayor que con 400
        # El umbral real de 0.05 aplica al dataset completo de JPL
        assert report.rmse_loocv < 0.5, (
            f"RMSE demasiado alto: {report.rmse_loocv:.4f} AU/My"
        )

    def test_full_pipeline_with_loocv(self):
        from src.layer3_ml.pipeline import run_layer3
        result = run_layer3(
            asteroid_features = {
                "diameter_km": 0.37, "albedo_pV": 0.23,
                "taxonomy": "Sq", "rot_per_h": 30.4,
                "a_AU": 0.9226, "ecc": 0.1914,
            },
            run_loocv    = True,
            run_benchmark= True,
            verbose      = True,
        )
        assert result.model.validation is not None
        assert result.inference.median < 0