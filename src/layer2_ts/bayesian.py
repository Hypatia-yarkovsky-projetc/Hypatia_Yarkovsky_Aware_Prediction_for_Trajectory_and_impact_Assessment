"""
bayesian.py
Actualización bayesiana para HYPATIA Capa 2.
Combina prior ML (Capa 3) con verosimilitud de regresión (OLS/HAC/STL).
Incluye límite físico obligatorio para evitar explosión numérica por ruido.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .regression import RegressionResult

@dataclass
class GaussianPrior:
    """Distribución gaussiana N(μ, σ²) para da/dt."""
    mean: float
    std: float
    source: str = "unknown"

    @classmethod
    def from_quantiles(cls, quantiles: dict, min_std: float = 0.05, source: str = "unknown") -> "GaussianPrior":
        q50 = quantiles[0.50]
        iqr = quantiles[0.75] - quantiles[0.25]
        std = max(iqr / 1.349, min_std)
        return cls(mean=q50, std=std, source=source)

    def summary(self) -> str:
        return f"Prior({self.source}): μ={self.mean:+.4f}, σ={self.std:.4f} AU/My"


@dataclass
class BayesianPosterior:
    """Resultado de la actualización bayesiana."""
    mean: float
    std: float
    prior: GaussianPrior
    data_mean: float = 0.0
    data_std: float = 0.0
    source: str = "bayes"

    def summary(self) -> str:
        return f"Posterior({self.source}): μ={self.mean:+.4f}, σ={self.std:.4f} AU/My"


def full_bayesian_estimation(reg_result: RegressionResult, ml_quantiles: dict, verbose: bool = True) -> BayesianPosterior:
    """Actualización bayesiana blindada con límite físico obligatorio."""
    prior = GaussianPrior.from_quantiles(ml_quantiles, source="ml_xgboost")
    
    data_mean = float(reg_result.dadt_au_my)
    data_std  = max(float(reg_result.std_error), 0.05)
    
    # 🔒 LÍMITE FÍSICO OBLIGATORIO
    # Yarkovsky en NEOs reales nunca supera ±1.0 AU/My.
    # Valores mayores indican que OLS ajustó ruido/offset, no la tendencia física.
    PHYSICAL_LIMIT = 1.0  # AU/My
    if abs(data_mean) > PHYSICAL_LIMIT:
        if verbose:
            print(f"[HYPATIA L2] ⚠ Regresión OLS arrojó valor no físico ({data_mean:.2f} AU/My). "
                  f"Clipeando a ±{PHYSICAL_LIMIT} y penalizando confianza.")
        data_mean = float(np.clip(data_mean, -PHYSICAL_LIMIT, PHYSICAL_LIMIT))
        data_std = max(data_std, 0.20)
        
    w_prior = 1.0 / (prior.std ** 2)
    w_data  = 1.0 / (data_std ** 2)
    
    post_mean = (prior.mean * w_prior + data_mean * w_data) / (w_prior + w_data)
    post_std  = np.sqrt(1.0 / (w_prior + w_data))
    
    if verbose:
        print(f"Posterior: μ={post_mean:+.4f} AU/My, σ={post_std:.4f} | "
              f"Peso Prior: {w_prior/(w_prior+w_data)*100:.0f}% | "
              f"Peso Datos: {w_data/(w_prior+w_data)*100:.0f}% | "
              f"IC 95%: [{post_mean-1.96*post_std:+.3f}, {post_mean+1.96*post_std:+.3f}]")
              
    return BayesianPosterior(
        mean=post_mean, std=post_std, prior=prior,
        data_mean=data_mean, data_std=data_std,
        source=f"{reg_result.method}+bayes"
    )