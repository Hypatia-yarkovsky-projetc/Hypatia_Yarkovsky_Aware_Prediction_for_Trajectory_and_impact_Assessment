"""
bayesian.py
Actualización bayesiana para HYPATIA Capa 2.
Combina prior ML (Capa 3) con verosimilitud de regresión (OLS/HAC/STL).
Incluye límite físico obligatorio para evitar explosión numérica por ruido.
"""
import numpy as np
import pandas as pd
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
        """Construye prior desde cuantiles ML o regresión."""
        q50 = quantiles[0.50]
        iqr = quantiles[0.75] - quantiles[0.25]
        # Conversión IQR → σ gaussiana, con piso de 0.05 AU/My
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


def bayesian_update(prior: GaussianPrior, data_mean: float, data_std: float, verbose: bool = False) -> BayesianPosterior:
    """
    Actualización bayesiana genérica para distribuciones gaussianas.
    Combina un prior N(μ0, σ0) con una verosimilitud N(μ1, σ1).
    """
    w_prior = 1.0 / (prior.std ** 2)
    w_data  = 1.0 / (data_std ** 2)
    
    post_mean = (prior.mean * w_prior + data_mean * w_data) / (w_prior + w_data)
    post_std  = np.sqrt(1.0 / (w_prior + w_data))
    
    if verbose:
        print(f"Update: μ={post_mean:+.4f}, σ={post_std:.4f}")
        
    return BayesianPosterior(
        mean=post_mean, std=post_std, prior=prior,
        data_mean=data_mean, data_std=data_std,
        source="generic_update"
    )


def full_bayesian_estimation(reg_result: RegressionResult, ml_quantiles: dict, verbose: bool = True) -> BayesianPosterior:
    """
    Actualización bayesiana blindada: prior ML + verosimilitud de regresión.
    Aplica límites físicos estrictos para evitar divergencia numérica.
    Incluye piso de peso para datos si hay señal moderada (SNR > 0.4).
    """
    prior = GaussianPrior.from_quantiles(ml_quantiles, source="ml_xgboost")
    
    data_mean = float(reg_result.dadt_au_my)
    data_std  = max(float(reg_result.std_error), 0.05)
    
    # LÍMITE FÍSICO OBLIGATORIO
    # El efecto Yarkovsky en NEOs reales nunca supera ±1.0 AU/My.
    # Valores mayores indican que la regresión ajustó ruido o un offset sistemático.
    PHYSICAL_LIMIT = 1.0  # AU/My
    
    if abs(data_mean) > PHYSICAL_LIMIT:
        if verbose:
            print(f"[HYPATIA L2] ⚠ Regresión OLS arrojó valor no físico ({data_mean:.2f} AU/My). "
                  f"Clipeando a ±{PHYSICAL_LIMIT} y penalizando confianza.")
        data_mean = float(np.clip(data_mean, -PHYSICAL_LIMIT, PHYSICAL_LIMIT))
        data_std = max(data_std, 0.20)  # Aumentar incertidumbre si el ajuste fue ruidoso
        
    # Calcular pesos bayesianos explícitamente para poder aplicar el ajuste de SNR
    w_prior = 1.0 / (prior.std ** 2)
    w_data  = 1.0 / (data_std ** 2)
    
    # PARCHE DE PESO MÍNIMO (Localizado y seguro)
    # Garantiza que los datos tengan al menos un 20% de influencia si hay señal detectable
    snr = abs(data_mean) / data_std
    if snr > 0.4:
        min_data_weight = 0.20
        total_w = w_prior + w_data
        if w_data / total_w < min_data_weight:
            w_data = w_prior * (min_data_weight / (1 - min_data_weight))
            
    # Cálculo directo del posterior (usa los pesos ya ajustados)
    post_mean = (prior.mean * w_prior + data_mean * w_data) / (w_prior + w_data)
    post_std  = np.sqrt(1.0 / (w_prior + w_data))
    
    posterior = BayesianPosterior(
        mean=post_mean, std=post_std, prior=prior,
        data_mean=data_mean, data_std=data_std,
        source=f"{reg_result.method}+bayes"
    )
    
    if verbose:
        print(f"Posterior: μ={posterior.mean:+.4f} AU/My, σ={posterior.std:.4f} | "
              f"Peso Prior: {w_prior/(w_prior+w_data)*100:.0f}% | "
              f"Peso Datos: {w_data/(w_prior+w_data)*100:.0f}% | "
              f"IC 95%: [{posterior.mean-1.96*posterior.std:+.3f}, {posterior.mean+1.96*posterior.std:+.3f}]")
              
    return posterior


def compare_posteriors_by_arc(
    reg_results: dict[float, RegressionResult],
    ml_quantiles: dict,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compara posteriors bayesianos para diferentes longitudes de arco observacional.
    Útil para análisis de sensibilidad y visualización en Capa 2.
    
    Args:
        reg_results: dict {años_de_arco: RegressionResult}
        ml_quantiles: prior ML {0.10: ..., 0.90: ...}
        verbose: imprimir tabla resumen
        
    Returns:
        DataFrame con columnas: arc_years, post_mean, post_std, weight_data_pct, ci_lower, ci_upper
    """
    rows = []
    for arc_years, reg in reg_results.items():
        post = full_bayesian_estimation(reg, ml_quantiles, verbose=False)
        
        w_prior = 1.0 / (post.prior.std ** 2)
        w_data  = 1.0 / (max(reg.std_error, 0.05) ** 2)
        
        rows.append({
            "arc_years": arc_years,
            "post_mean": post.mean,
            "post_std": post.std,
            "weight_data_pct": (w_data / (w_prior + w_data)) * 100,
            "ci_lower": post.mean - 1.96 * post.std,
            "ci_upper": post.mean + 1.96 * post.std,
        })
        
    df = pd.DataFrame(rows).sort_values("arc_years").reset_index(drop=True)
    
    if verbose and not df.empty:
        print("\n[HYPATIA L2] Comparación de posteriors por arco observacional:")
        print(df.to_string(index=False, float_format="{:.4f}".format))
        
    return df