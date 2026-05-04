"""
features.py
Ingeniería de features para el modelo ML de HYPATIA.
Cada feature tiene justificación física directa con el efecto Yarkovsky.
Corrección de sesgo mediante Inverse Density Weighting (IDW).
"""
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

FEATURE_MAP = {
    "inv_diameter": "inv_diameter",
    "absorptivity": "absorptivity",
    "tax_code"    : "tax_code",
    "rot_per_h"   : "rot_per_h",
    "a_AU"        : "a_AU",
    "ecc"         : "ecc",
}
FEATURE_NAMES = list(FEATURE_MAP.values())
TARGET_NAME   = "dadt_AuMy"

FEATURE_DESCRIPTIONS = {
    "inv_diameter": "Inverso del diámetro (1/km) — Yarkovsky ∝ 1/D",
    "absorptivity": "Absortividad (1 − albedo) — energía solar absorbida",
    "tax_code"    : "Clase taxonómica (numérico) — proxy conductividad térmica",
    "rot_per_h"   : "Período de rotación (h) — desfase térmico diurno",
    "a_AU"        : "Semieje mayor (AU) — intensidad irradiación ∝ 1/a²",
    "ecc"         : "Excentricidad — variación temporal del calentamiento",
}

FEATURE_BOUNDS = {
    "inv_diameter": (1/500.0, 1/0.001),
    "absorptivity": (0.0, 0.99),
    "tax_code"    : (0.0, 1.0),
    "rot_per_h"   : (0.1, 1000.0),
    "a_AU"        : (0.1, 5.0),
    "ecc"         : (0.0, 0.99),
}

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae y valida el vector de features del DataFrame de entrenamiento."""
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Columnas de features faltantes: {missing}")

    X = df[FEATURE_NAMES].copy()

    for feat, (lo, hi) in FEATURE_BOUNDS.items():
        out_of_range = (X[feat] < lo) | (X[feat] > hi)
        if out_of_range.any():
            n = out_of_range.sum()
            print(f"[HYPATIA L3] ⚠ {n} valores de '{feat}' fuera de "
                  f"rango [{lo:.3g}, {hi:.3g}] — clipeando")
            X[feat] = X[feat].clip(lo, hi)

    return X

def build_feature_vector(
    diameter_km    : float,
    albedo_pV      : float,
    taxonomy       : str,
    rot_per_h      : float,
    a_AU           : float,
    ecc            : float,
) -> dict:
    """Construye el vector de features para un único asteroide nuevo."""
    from .dataset import TAX_CODE

    tax_first = taxonomy.strip()[0].upper() if taxonomy.strip() else "S"
    tax_c = TAX_CODE.get(tax_first, TAX_CODE.get(taxonomy.strip(), 0.5))

    features = {
        "inv_diameter": 1.0 / max(diameter_km, 0.001),
        "absorptivity": max(0.0, min(0.99, 1.0 - albedo_pV)),
        "tax_code"    : float(tax_c),
        "rot_per_h"   : max(0.1, rot_per_h),
        "a_AU"        : max(0.1, a_AU),
        "ecc"         : max(0.0, min(0.99, ecc)),
    }
    return features

def compute_sample_weights(
    X           : pd.DataFrame,
    method      : str = "kde",
    bandwidth   : float = 0.5,
    normalize   : bool = True,
) -> np.ndarray:
    """Calcula pesos de muestra para corregir el sesgo de observabilidad."""
    if method == "uniform":
        return np.ones(len(X))

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X.values)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(X_std)

    log_density = kde.score_samples(X_std)
    density     = np.exp(log_density)
    density     = np.maximum(density, 1e-10)

    weights = 1.0 / density
    cap = np.percentile(weights, 99)
    weights = np.minimum(weights, cap)

    if normalize:
        weights = weights / weights.mean()

    return weights.astype(np.float32)

def validate_new_asteroid(features: dict) -> tuple[bool, list[str]]:
    """Valida las features de un asteroide nuevo antes de la predicción."""
    warnings_list = []

    for feat, (lo, hi) in FEATURE_BOUNDS.items():
        val = features.get(feat)
        if val is None:
            warnings_list.append(f"Feature '{feat}' faltante")
            continue
        if val < lo or val > hi:
            warnings_list.append(
                f"'{feat}' = {val:.4g} fuera del rango de entrenamiento  "
                f"[{lo:.3g}, {hi:.3g}]"
            )

    inv_d = features.get("inv_diameter", 0)
    if inv_d > 1.0:
        warnings_list.append(
            "Diámetro < 1 km: extrapolación probable, incertidumbre mayor"
        )

    valid = len([w for w in warnings_list if "faltante" in w]) == 0
    return valid, warnings_list

def feature_importance_names() -> list[str]:
    return FEATURE_NAMES.copy()

def features_to_dataframe(features_dict: dict) -> pd.DataFrame:
    return pd.DataFrame([{f: features_dict[f] for f in FEATURE_NAMES}])