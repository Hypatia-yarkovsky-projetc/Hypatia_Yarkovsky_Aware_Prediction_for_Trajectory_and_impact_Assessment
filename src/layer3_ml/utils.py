"""
utils.py
Utilidades Capa 3: verificación de integración, diagnóstico y benchmark.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from .model import HypatiaModel, QUANTILES
from .features import FEATURE_NAMES

def verify_layer3_integration() -> bool:
    """Verifica importaciones y contratos con Capas 1 y 2."""
    errors = []
    try:
        from ..layer1_ode.yarkovsky import yarkovsky_order_of_magnitude, dadt_to_A2
        from ..layer1_ode.constants import GM_SOL_AU3_DAY2
        A2 = dadt_to_A2(-0.20, 0.9226, 0.1914)
        assert abs(A2) > 0, "A2 de Apophis no puede ser cero"
    except Exception as e:
        errors.append(f"Capa 1: {e}")

    try:
        from ..layer2_ts.bayesian import GaussianPrior
        prior = GaussianPrior.from_quantiles({0.10: -0.28, 0.25: -0.24, 0.50: -0.19, 0.75: -0.15, 0.90: -0.11})
        assert prior.std > 0, "sigma del prior debe ser positivo"
    except Exception as e:
        errors.append(f"Capa 2: {e}")

    if errors:
        for e in errors:
            print(f"[HYPATIA L3] Error: {e}")
        return False
    print("[HYPATIA L3] Integracion con Capas 1 y 2 verificada")
    return True

def verify_output_format(model: HypatiaModel) -> bool:
    """Verifica que la salida coincide con el contrato de la Capa 2."""
    from .inference import inferir_dadt
    test_result = inferir_dadt(
        diameter_km=0.37, albedo_pV=0.23, taxonomy="Sq",
        rot_per_h=30.4, a_AU=0.9226, ecc=0.1914, model=model, verbose=False
    )
    prior_dict = test_result.to_layer2_prior()
    required_keys = {0.10, 0.25, 0.50, 0.75, 0.90}
    missing = required_keys - set(prior_dict.keys())
    if missing:
        print(f"[HYPATIA L3] Claves faltantes en prior: {missing}")
        return False

    vals = [prior_dict[q] for q in sorted(prior_dict)]
    if not all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
        print("[HYPATIA L3] Cuantiles no monotonos")
        return False
    print("[HYPATIA L3] Formato de salida verificado — compatible con Capa 2")
    return True

def model_summary(model: HypatiaModel) -> str:
    """Resumen de hiperparámetros y validación."""
    lines = [
        "=" * 52,
        "  HYPATIA L3 — Modelo de inferencia Yarkovsky",
        "=" * 52,
        f"  Entrenado     : {'Si' if model.is_fitted else 'No'}",
        f"  N muestras    : {model.n_training}",
        f"  Cuantiles     : {[f'Q{int(q*100)}' for q in sorted(model.quantile_models)]}",
        f"  Features      : {model.feature_names}",
        "  Hiperparametros:",
    ]
    for k, v in model.params.items():
        if k not in ("verbosity", "random_state", "tree_method"):
            lines.append(f"    {k:<20}: {v}")
    if model.validation is not None:
        lines.append("")
        lines.append(model.validation.summary())
    else:
        lines.append("  Validacion LOO-CV: no ejecutada")
    lines.append("=" * 52)
    return "\n".join(lines)

def check_monotonicity(model: HypatiaModel, df: pd.DataFrame) -> dict:
    """Verifica monotonicidad de cuantiles predichos."""
    from .features import extract_features
    X = extract_features(df)
    preds = model.predict_quantiles(X)
    violations = 0
    max_violation = 0.0

    for i in range(len(X)):
        for j in range(len(QUANTILES) - 1):
            q1, q2 = QUANTILES[j], QUANTILES[j+1]
            diff = preds[q2][i] - preds[q1][i]
            if diff < 0:
                violations += 1
                max_violation = max(max_violation, abs(diff))

    rate = violations / (len(X) * (len(QUANTILES) - 1))
    print(f"[HYPATIA L3] Monotonicidad: {violations} violaciones ({rate:.1%}), max={max_violation:.4f} AU/My")
    return {"violations": violations, "violation_rate": rate, "max_violation": max_violation}

REFERENCE_ASTEROIDS = {
    "Apophis": {"diameter_km": 0.37, "albedo_pV": 0.23, "taxonomy": "Sq", "rot_per_h": 30.4, "a_AU": 0.9226, "ecc": 0.1914, "true_dadt": -0.200},
    "Bennu": {"diameter_km": 0.49, "albedo_pV": 0.044, "taxonomy": "B", "rot_per_h": 4.3, "a_AU": 1.1264, "ecc": 0.2037, "true_dadt": -0.284},
    "Ryugu": {"diameter_km": 0.90, "albedo_pV": 0.045, "taxonomy": "C", "rot_per_h": 7.63, "a_AU": 1.1896, "ecc": 0.1902, "true_dadt": -0.076},
    "Itokawa": {"diameter_km": 0.32, "albedo_pV": 0.53, "taxonomy": "S", "rot_per_h": 12.13, "a_AU": 1.3241, "ecc": 0.2801, "true_dadt": -0.094},
}

def benchmark_reference_asteroids(model: HypatiaModel, verbose: bool = True) -> pd.DataFrame:
    """Evalua modelo sobre asteroides de referencia con da/dt conocido."""
    from .inference import inferir_dadt
    rows = []
    for name, props in REFERENCE_ASTEROIDS.items():
        props_copy = props.copy()
        true_dadt = props_copy.pop("true_dadt")
        res = inferir_dadt(**props_copy, model=model, verbose=False)
        error = abs(res.median - true_dadt)
        in_ci = res.ci_80_lower <= true_dadt <= res.ci_80_upper
        rows.append({
            "Asteroide": name, "da/dt real": true_dadt, "Q50 (pred)": res.median,
            "IC 80%": f"[{res.ci_80_lower:.3f}, {res.ci_80_upper:.3f}]",
            "Error abs": error, "En IC 80%": "Si" if in_ci else "No",
        })
    df = pd.DataFrame(rows)
    if verbose:
        print("\n[HYPATIA L3] Benchmark sobre asteroides de referencia:")
        print(df.to_string(index=False))
        covered = df["En IC 80%"].eq("Si").sum()
        print(f"\n  {covered}/{len(df)} asteroides dentro del IC 80%")
    return df

def save_model(model: HypatiaModel, path: str) -> None:
    model.save(path)

def load_model(path: str) -> HypatiaModel:
    return HypatiaModel.load(path)

def get_apophis_prior(model: HypatiaModel) -> dict:
    from .inference import apophis_inference
    return apophis_inference(model).to_layer2_prior()