"""
layer3_ml
Capa 3 de HYPATIA: Inferencia del parámetro Yarkovsky mediante ML.
Infiere da/dt para asteroides recién descubiertos usando XGBoost
cuantílico entrenado en ~400 asteroides con Yarkovsky medido.
Salida: distribución P(da/dt|features) → Prior bayesiano para Capa 2.
"""
# Pipeline
from .pipeline import run_layer3, run_layer3_offline, Layer3Result

# Dataset
from .dataset import (
    build_training_dataset,
    load_training_dataset,
    download_yarkovsky_sbdb,
    compute_dadt_from_A2,
    TAX_CODE,
    TAX_ALBEDO_MEDIAN,
)

# Features
from .features import (
    extract_features,
    build_feature_vector,
    compute_sample_weights,
    validate_new_asteroid,
    features_to_dataframe,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    FEATURE_BOUNDS,
    TARGET_NAME,
)

# Modelo
from .model import (
    train,
    validate_loocv,
    attach_validation,
    HypatiaModel,
    ValidationReport,
    QUANTILES,
    DEFAULT_PARAMS,
)

# Inferencia
from .inference import (
    inferir_dadt,
    inferir_dadt_batch,
    apophis_inference,
    InferenceResult,
)

# Utilidades
from .utils import (
    verify_layer3_integration,
    verify_output_format,
    model_summary,
    check_monotonicity,
    benchmark_reference_asteroids,
    save_model,
    load_model,
    get_apophis_prior,
    REFERENCE_ASTEROIDS,
)

# Visualización
from .visualizer import (
    plot_prediction_distribution,
    plot_feature_importance,
    plot_loocv_scatter,
    plot_dataset_distribution,
    plot_quantile_calibration,
)

__all__ = [
    "run_layer3", "run_layer3_offline", "Layer3Result",
    "build_training_dataset", "load_training_dataset",
    "download_yarkovsky_sbdb", "compute_dadt_from_A2",
    "TAX_CODE", "TAX_ALBEDO_MEDIAN",
    "extract_features", "build_feature_vector", "compute_sample_weights",
    "validate_new_asteroid", "features_to_dataframe",
    "FEATURE_NAMES", "FEATURE_DESCRIPTIONS", "FEATURE_BOUNDS", "TARGET_NAME",
    "train", "validate_loocv", "attach_validation",
    "HypatiaModel", "ValidationReport", "QUANTILES", "DEFAULT_PARAMS",
    "inferir_dadt", "inferir_dadt_batch", "apophis_inference", "InferenceResult",
    "verify_layer3_integration", "verify_output_format", "model_summary",
    "check_monotonicity", "benchmark_reference_asteroids",
    "save_model", "load_model", "get_apophis_prior", "REFERENCE_ASTEROIDS",
    "plot_prediction_distribution", "plot_feature_importance",
    "plot_loocv_scatter", "plot_dataset_distribution", "plot_quantile_calibration",
]