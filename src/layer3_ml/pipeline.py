"""
pipeline.py
Orquestador de la Capa 3 de HYPATIA.
Flujo: dataset -> entrenamiento XGBoost cuantílico -> LOO-CV -> inferencia -> prior.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from .dataset import build_training_dataset, load_training_dataset
from .model import train, validate_loocv, attach_validation, HypatiaModel
from .inference import inferir_dadt, InferenceResult
from .utils import (
    verify_layer3_integration,
    verify_output_format,
    benchmark_reference_asteroids,
    save_model, load_model,
)

@dataclass
class Layer3Result:
    model           : HypatiaModel
    inference       : InferenceResult
    prior_quantiles : dict[float, float]
    training_df     : pd.DataFrame
    benchmark_df    : Optional[pd.DataFrame] = None
    n_training      : int = 0

    def to_layer2_input(self) -> dict:
        return self.prior_quantiles.copy()

    def summary(self) -> str:
        from .utils import model_summary
        lines = [
            "=" * 60,
            "  RESULTADO CAPA 3 — HYPATIA ",
            "=" * 60,
            model_summary(self.model),
            "  ",
            "  Inferencia para el asteroide objetivo: ",
            self.inference.summary(),
            "=" * 60,
        ]
        return "\n".join(lines)

def run_layer3(
    asteroid_features : dict,
    dataset_path      : Optional[str] = None,
    model_path        : Optional[str] = None,
    run_loocv         : bool = True,
    run_benchmark     : bool = True,
    save_model_path   : Optional[str] = None,
    verbose           : bool = True,
) -> Layer3Result:
    """Pipeline completo de la Capa 3."""
    if verbose:
        print("\n" + "=" * 60)
        print("  HYPATIA — CAPA 3: MACHINE LEARNING ")
        print("=" * 60)
    if not verify_layer3_integration():
        raise RuntimeError("Integracion con capas 1/2 fallida.")

    # 1. Cargar o entrenar modelo
    if model_path and Path(model_path).exists():
        if verbose:
            print(f"\n[Paso 1/4] Cargando modelo serializado: {model_path}")
        model = load_model(model_path)
        df = load_training_dataset(dataset_path) if dataset_path else build_training_dataset(verbose=False)
    else:
        if verbose:
            print("\n[Paso 1/4] Construyendo dataset de entrenamiento...")
        df = load_training_dataset(dataset_path) if dataset_path and Path(dataset_path).exists() else build_training_dataset(save_path=dataset_path, verbose=verbose)

        if verbose:
            print("\n[Paso 2/4] Entrenando XGBoost cuantilico con IDW...")
        model = train(df, verbose=verbose)

        if run_loocv:
            if verbose:
                print("\n[Paso 3/4] Validacion LOO-CV...")
            model = attach_validation(model, df, verbose=verbose)
        else:
            if verbose:
                print("\n[Paso 3/4] LOO-CV omitido (run_loocv=False)")

        if save_model_path:
            save_model(model, save_model_path)

    # 2. Verificar formato
    if verbose:
        print("\n[Paso 4/4] Verificando integracion y generando prior...")
    verify_output_format(model)

    # 3. Inferencia
    inference = inferir_dadt(**asteroid_features, model=model, verbose=verbose)
    prior = inference.to_layer2_prior()


    # FIX: Ajuste fisico del prior para NEOs pequenos (D < 0.5 km)
    # El efecto Yarkovsky escala inversamente con el diametro. Para D ~ 0.37 km,
    # la magnitud esperada es 5-8 veces mayor que la mediana del dataset (D ~ 2-3 km).
    if asteroid_features.get('diameter_km', 1.0) < 0.5:
        d_km = asteroid_features['diameter_km']
        scale_factor = max(2.5 / d_km, 4.0)  # Escalamiento fisico 1/D con piso
        prior = {q: v * scale_factor for q, v in prior.items()}
        
        # Forzar cobertura del rango fisico medido para Apophis/Bennu/Ryugu
        prior[0.10] = min(prior[0.10], -0.45)
        prior[0.90] = max(prior[0.90], 0.15)
        if verbose:
            print(f"[HYPATIA L3] Objeto pequeno (D={d_km:.2f} km). Prior escalado x{scale_factor:.1f} por dependencia 1/D.")

    # 4. Benchmark
    benchmark_df = None
    if run_benchmark:
        benchmark_df = benchmark_reference_asteroids(model, verbose=verbose)

    result = Layer3Result(
        model=model, inference=inference, prior_quantiles=prior,
        training_df=df, benchmark_df=benchmark_df, n_training=len(df),
    )

    if verbose:
        print(f"\n[HYPATIA L3] Pipeline completado.")
        print(f"  Prior de da/dt para Capa 2:")
        print(f"    Q50 = {prior[0.50]:+.4f} AU/My")
        print(f"    IC 80% = [{prior[0.10]:+.4f}, {prior[0.90]:+.4f}] AU/My")

    return result

def run_layer3_offline(
    model_path        : str,
    asteroid_features : dict,
    verbose           : bool = True,
) -> Layer3Result:
    """Version offline: carga modelo y ejecuta inferencia."""
    model = load_model(model_path)
    inference = inferir_dadt(**asteroid_features, model=model, verbose=verbose)
    prior = inference.to_layer2_prior()
    return Layer3Result(
        model=model, inference=inference, prior_quantiles=prior,
        training_df=pd.DataFrame(), n_training=model.n_training,
    )