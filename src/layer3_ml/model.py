"""
model.py
Modelo de inferencia del parámetro Yarkovsky da/dt para HYPATIA.
Arquitectura: XGBoost con regresión cuantílica (objective='reg:quantileerror').
Un modelo por cuantil: Q10, Q25, Q50, Q75, Q90.
Validación: Leave-One-Out Cross Validation (LOO-CV).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import joblib
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .features import (
    extract_features, compute_sample_weights,
    FEATURE_NAMES, TARGET_NAME
)
warnings.filterwarnings("ignore", category=UserWarning)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

DEFAULT_PARAMS = {
    "max_depth"       : 3,
    "n_estimators"    : 200,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
    "min_child_weight": 3,
    "random_state"    : 42,
    "tree_method"     : "hist",
    "verbosity"       : 0,
}

@dataclass
class ValidationReport:
    """Resultados de la validación LOO-CV del modelo."""
    rmse_loocv   : float
    mae_loocv    : float
    r2_loocv     : float
    y_true       : np.ndarray
    y_pred_loocv : np.ndarray
    n_samples    : int
    feature_names: list[str]
    importances  : np.ndarray

    @property
    def passed(self) -> bool:
        return self.rmse_loocv < 0.05

    def summary(self) -> str:
        status = "✓ APROBADO" if self.passed else "⚠ REVISAR"
        lines = [
            f"  Validación LOO-CV — {status}",
            f"  {'─'*44}",
            f"  RMSE LOO-CV : {self.rmse_loocv:.4f} AU/My   "
              f"(meta: < 0.05)",
            f"  MAE LOO-CV  : {self.mae_loocv:.4f} AU/My",
            f"  R² LOO-CV   : {self.r2_loocv:.4f}",
            f"  N muestras  : {self.n_samples}",
            f"  {'─'*44}",
            "  Importancia de features (top 3):",
        ]
        imp_sorted = sorted(
            zip(self.feature_names, self.importances),
            key=lambda x: x[1], reverse=True
        )
        for name, imp in imp_sorted[:3]:
            lines.append(f"    {name:<16}: {imp:.3f}")
        return "\n".join(lines)

@dataclass
class HypatiaModel:
    """Contenedor del modelo entrenado de HYPATIA Capa 3."""
    quantile_models : dict[float, XGBRegressor] = field(default_factory=dict)
    median_model    : Optional[XGBRegressor] = None
    sample_weights  : Optional[np.ndarray] = None
    validation       : Optional[ValidationReport] = None
    feature_names   : list[str] = field(default_factory=lambda: FEATURE_NAMES.copy())
    n_training      : int = 0
    is_fitted       : bool = False 
    params          : dict = field(default_factory=lambda: DEFAULT_PARAMS.copy())

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Llama a train() primero.")

        X_vals = X[self.feature_names].values
        return {
            q: self.quantile_models[q].predict(X_vals)
            for q in QUANTILES
        }

    def predict_single(self, features: dict) -> dict[float, float]:
        from .features import features_to_dataframe
        X = features_to_dataframe(features)
        preds = self.predict_quantiles(X)
        return {q: float(arr[0]) for q, arr in preds.items()}

    def to_layer2_prior(self, features: dict) -> dict:
        return self.predict_single(features)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[HYPATIA L3] Modelo guardado: {path}")

    @classmethod
    def load(cls, path: str) -> "HypatiaModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        model = joblib.load(path)
        print(f"[HYPATIA L3] Modelo cargado: {path}  "
              f"({model.n_training} muestras de entrenamiento)")
        return model

def train(
    df          : pd.DataFrame,
    params      : Optional[dict] = None,
    quantiles   : list[float] = QUANTILES,
    use_weights : bool = True,
    verbose     : bool = True,
) -> HypatiaModel:
    """Entrena el modelo XGBoost cuantílico de HYPATIA."""
    _params = {**DEFAULT_PARAMS, **(params or {})}

    X = extract_features(df)
    y = df[TARGET_NAME].values

    weights = (
        compute_sample_weights(X) if use_weights
        else np.ones(len(X), dtype=np.float32)
    )

    if verbose:
        print(f"\n[HYPATIA L3] Entrenando modelo XGBoost cuantílico")
        print(f"  Muestras      : {len(X)}")
        print(f"  Features      : {FEATURE_NAMES}")
        print(f"  Cuantiles     : {quantiles}")
        print(f"  Pesos IDW     : {'Sí' if use_weights else 'No'}")

    quantile_models = {}
    for q in quantiles:
        model = XGBRegressor(
            objective       = "reg:quantileerror",
            quantile_alpha  = q,
            **{k: v for k, v in _params.items()
               if k not in ("objective", "quantile_alpha")},
        )
        model.fit(X.values, y, sample_weight=weights)
        quantile_models[q] = model
        if verbose:
            print(f"  Q{q*100:.0f} entrenado ", end="|  ", flush=True)

    if verbose:
        print("\n[HYPATIA L3] Todos los cuantiles entrenados.")

    median_model = XGBRegressor(
        objective="reg:squarederror",
        **{k: v for k, v in _params.items()
           if k not in ("objective", "quantile_alpha")},
    )
    median_model.fit(X.values, y, sample_weight=weights)

    hypatia_model = HypatiaModel(
        quantile_models = quantile_models,
        median_model    = median_model,
        sample_weights  = weights,
        feature_names   = FEATURE_NAMES,
        n_training      = len(X),
        is_fitted       = True,
        params          = _params,
    )

    return hypatia_model

def validate_loocv(
    df      : pd.DataFrame,
    params  : Optional[dict] = None,
    verbose : bool = True,
) -> ValidationReport:
    """Valida el modelo mediante Leave-One-Out Cross Validation."""
    _params = {**DEFAULT_PARAMS, **(params or {})}

    X = extract_features(df)
    y = df[TARGET_NAME].values
    weights = compute_sample_weights(X)

    if verbose:
        print(f"\n[HYPATIA L3] LOO-CV sobre {len(X)} muestras...")
        print("  (esto tarda ~2-5 minutos dependiendo del hardware)")

    model_for_cv = XGBRegressor(
        objective="reg:squarederror",
        **{k: v for k, v in _params.items()
           if k not in ("objective",)},
    )

    loo = LeaveOneOut()
    y_pred = cross_val_predict(
        model_for_cv,
        X.values, y,
        cv=loo,
        fit_params={"sample_weight": weights},
        n_jobs=1,
    )

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae  = float(mean_absolute_error(y, y_pred))
    r2   = float(r2_score(y, y_pred))

    full_model = XGBRegressor(
        objective="reg:squarederror",
        **{k: v for k, v in _params.items() if k not in ("objective",)},
    )
    full_model.fit(X.values, y, sample_weight=weights)
    importances = full_model.feature_importances_

    report = ValidationReport(
        rmse_loocv    = rmse,
        mae_loocv     = mae,
        r2_loocv      = r2,
        y_true        = y,
        y_pred_loocv  = y_pred,
        n_samples     = len(y),
        feature_names = FEATURE_NAMES,
        importances   = importances,
    )

    if verbose:
        print(report.summary())

    return report

def attach_validation(
    model   : HypatiaModel,
    df      : pd.DataFrame,
    params  : Optional[dict] = None,
    verbose : bool = True,
) -> HypatiaModel:
    """Corre LOO-CV y adjunta el ValidationReport al modelo."""
    report = validate_loocv(df, params=params or model.params, verbose=verbose)
    model.validation = report
    return model