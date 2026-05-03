"""
inference.py
Interfaz de inferencia de la Capa 3: predice da/dt para un asteroide
nuevo usando el modelo XGBoost cuantílico entrenado.
Este módulo es el punto de salida hacia la Capa 2. Su función principal
inferir_dadt() recibe las propiedades físicas observables y retorna
la distribución P(da/dt | features) como dict de cuantiles.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from .features import build_feature_vector, validate_new_asteroid
from .model import HypatiaModel, QUANTILES
from .dataset import TAX_CODE

@dataclass
class InferenceResult:
    """Resultado completo de la inferencia de da/dt para un asteroide."""
    quantiles     : dict[float, float]
    features_used : dict
    median         : float
    iqr           : float
    ci_80_lower   : float
    ci_80_upper   : float
    in_domain     : bool
    warnings      : list[str] = field(default_factory=list)
    physical_estimate : Optional[float] = None

    @property
    def uncertainty_au_my(self) -> float:
        return (self.ci_80_upper - self.ci_80_lower) / 2.0

    @property
    def sign(self) -> str:
        return "negativo" if self.median < 0 else "positivo"

    def to_layer2_prior(self) -> dict:
        return self.quantiles.copy()

    def summary(self) -> str:
        lines = [
            "  Inferencia da/dt — HYPATIA Capa 3",
            f"  {'─'*44}",
            f"  Mediana (Q50) : {self.median:+.4f} AU/My  [{self.sign}]",
            f"  IC 80%        : [{self.ci_80_lower:+.4f}, {self.ci_80_upper:+.4f}] AU/My",
            f"  Incertidumbre : ±{self.uncertainty_au_my:.4f} AU/My",
            f"  En dominio    : {'Sí' if self.in_domain else 'No (extrapolación)'}",
        ]
        if self.physical_estimate is not None:
            lines.append(
                f"  Est. física   : ±{self.physical_estimate:.4f} AU/My  "
                f"(orden de magnitud esperado)"
            )
        if self.warnings:
            lines.append("  Advertencias  : ")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append(f"  {'─'*44}")
        lines.append("  Distribución cuantílica: ")
        for q, v in self.quantiles.items():
            lines.append(f"    P{int(q*100):02d}: {v:+.4f} AU/My")
        return "\n".join(lines)

def inferir_dadt(
    diameter_km    : float,
    albedo_pV      : float,
    taxonomy       : str,
    rot_per_h      : float,
    a_AU           : float,
    ecc            : float,
    model          : HypatiaModel,
    validate_domain: bool = True,
    verbose        : bool = False,
) -> InferenceResult:
    """Predice da/dt para un asteroide nuevo. Interfaz limpia hacia Capa 2."""
    if not model.is_fitted:
        raise RuntimeError("Modelo no entrenado. Usa train() primero.")

    features = build_feature_vector(
        diameter_km, albedo_pV, taxonomy, rot_per_h, a_AU, ecc
    )

    warnings_list = []
    in_domain = True
    if validate_domain:
        in_domain, warnings_list = validate_new_asteroid(features)

    quantiles_pred = model.predict_single(features)

    from ..layer1_ode.yarkovsky import yarkovsky_order_of_magnitude
    try:
        phys_est = yarkovsky_order_of_magnitude(diameter_km, a_AU)
    except Exception:
        phys_est = None

    if phys_est is not None:
        ratio = abs(quantiles_pred[0.50]) / max(phys_est, 1e-6)
        if ratio > 10 or ratio < 0.1:
            warnings_list.append(
                f"Predicción ({quantiles_pred[0.50]:.3f} AU/My) difiere  "
                f"10× de la estimación física ({phys_est:.3f} AU/My)"
            )

    result = InferenceResult(
        quantiles         = quantiles_pred,
        features_used     = features,
        median            = quantiles_pred[0.50],
        iqr                = quantiles_pred[0.75] - quantiles_pred[0.25],
        ci_80_lower       = quantiles_pred[0.10],
        ci_80_upper       = quantiles_pred[0.90],
        in_domain         = in_domain,
        warnings          = warnings_list,
        physical_estimate = phys_est,
    )

    if verbose:
        print(result.summary())

    return result

def inferir_dadt_batch(
    df    : pd.DataFrame,
    model : HypatiaModel,
    verbose: bool = True,
) -> pd.DataFrame:
    """Predice da/dt para un DataFrame de múltiples asteroides."""
    if not model.is_fitted:
        raise RuntimeError("Modelo no entrenado.")

    results = []
    n = len(df)

    for i, row in df.iterrows():
        try:
            res = inferir_dadt(
                diameter_km = float(row.get("diameter_km", 1.0)),
                albedo_pV   = float(row.get("albedo_pV", row.get("albedo", 0.15))),
                taxonomy    = str(row.get("taxonomy", row.get("spec_T", "S"))),
                rot_per_h   = float(row.get("rot_per_h", row.get("rot_per", 7.0))),
                a_AU        = float(row.get("a_AU", row.get("a", 1.0))),
                ecc         = float(row.get("ecc", row.get("e", 0.2))),
                model       = model,
                validate_domain = False,
                verbose     = False,
            )
            results.append({
                "dadt_q10"        : res.quantiles[0.10],
                "dadt_q25"        : res.quantiles[0.25],
                "dadt_q50"        : res.quantiles[0.50],
                "dadt_q75"        : res.quantiles[0.75],
                "dadt_q90"        : res.quantiles[0.90],
                "dadt_uncertainty": res.uncertainty_au_my,
            })
        except Exception as e:
            results.append({k: np.nan for k in [
                "dadt_q10", "dadt_q25", "dadt_q50",
                "dadt_q75", "dadt_q90", "dadt_uncertainty"
            ]})

        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} asteroides procesados")

    result_df = pd.concat([df.reset_index(drop=True),
                           pd.DataFrame(results)], axis=1)

    if verbose:
        valid = result_df["dadt_q50"].notna().sum()
        print(f"[HYPATIA L3] Inferencia batch: {valid}/{n} exitosas")

    return result_df

def apophis_inference(model: HypatiaModel) -> InferenceResult:
    """Caso de referencia: infiere da/dt para Apophis con propiedades reales."""
    return inferir_dadt(
        diameter_km = 0.37,
        albedo_pV   = 0.23,
        taxonomy    = "Sq",
        rot_per_h   = 30.4,
        a_AU        = 0.9226,
        ecc         = 0.1914,
        model       = model,
        verbose     = True,
    )