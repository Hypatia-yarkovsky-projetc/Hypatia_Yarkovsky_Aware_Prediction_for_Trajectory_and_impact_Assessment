"""
bayesian.py
Actualización bayesiana analítica (conjugación Gaussiana).
Combina Prior (Capa 3) y Likelihood (Capa 2).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm
from .regression import RegressionResult

@dataclass
class GaussianPrior:
    """Prior gaussiano derivado de cuantiles ML."""
    mean: float; std: float; source: str = "ml_quantile"

    @classmethod
    def from_quantiles(cls, quantiles: dict, source: str = "ml_quantile") -> "GaussianPrior":
        mu = float(quantiles[0.50])
        if 0.10 in quantiles and 0.90 in quantiles:
            sigma = (float(quantiles[0.90]) - float(quantiles[0.10])) / (2 * 1.2816)
        elif 0.25 in quantiles and 0.75 in quantiles:
            sigma = (float(quantiles[0.75]) - float(quantiles[0.25])) / (2 * 0.6745)
        else:
            sigma = abs(mu) * 0.5
        return cls(mean=mu, std=max(sigma, 1e-6), source=source)

    @classmethod
    def uninformative(cls) -> "GaussianPrior":
        return cls(mean=0.0, std=1e6, source="uninformative")

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return norm.pdf(x, self.mean, self.std)

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        return np.random.default_rng(seed).normal(self.mean, self.std, n)

@dataclass
class BayesianPosterior:
    """Posterior resultante de la actualización."""
    mean: float; std: float; ci_lower: float; ci_upper: float
    prior: GaussianPrior; likelihood: RegressionResult
    weight_prior: float; weight_likelihood: float
    samples: np.ndarray = field(default_factory=lambda: np.array([]))
    n_samples: int = 0

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def sample_dadt(self, n: int = 1000, seed: int = 42) -> np.ndarray:
        return np.random.default_rng(seed).normal(self.mean, self.std, n)

    def to_layer1_input(self) -> dict:
        return {
            "dadt_mean": self.mean, "dadt_std": self.std,
            "ci_lower": self.ci_lower, "ci_upper": self.ci_upper,
            "samples": self.sample_dadt(200),
        }

    def summary(self) -> str:
        return (
            f"Posterior: μ={self.mean:+.4f} AU/My, σ={self.std:.4f} | "
            f"Peso Prior: {self.weight_prior:.0%} | "
            f"Peso Datos: {self.weight_likelihood:.0%} | "
            f"IC 95%: [{self.ci_lower:+.3f}, {self.ci_upper:+.3f}]"
        )

def bayesian_update(prior: GaussianPrior, likelihood: RegressionResult, n_samples: int = 2000, alpha: float = 0.05) -> BayesianPosterior:
    """Actualización bayesiana gaussiana conjugada."""
    tau_prior = 1.0 / (prior.std ** 2)
    tau_data = 1.0 / (likelihood.std_error ** 2) if likelihood.std_error > 1e-10 else 0.0
    tau_post = tau_prior + tau_data
    
    mu_post = (tau_prior * prior.mean + tau_data * likelihood.dadt_au_my) / tau_post
    std_post = 1.0 / np.sqrt(tau_post)
    
    z = norm.ppf(1 - alpha / 2)
    
    return BayesianPosterior(
        mean=float(mu_post), std=float(std_post),
        ci_lower=float(mu_post - z * std_post), ci_upper=float(mu_post + z * std_post),
        prior=prior, likelihood=likelihood,
        weight_prior=float(tau_prior / tau_post), weight_likelihood=float(tau_data / tau_post),
        samples=np.random.default_rng(42).normal(mu_post, std_post, n_samples),
        n_samples=n_samples,
    )

def full_bayesian_estimation(regression_result: RegressionResult, ml_quantiles: Optional[dict] = None, n_samples: int = 2000, verbose: bool = True) -> BayesianPosterior:
    """Función principal de estimación bayesiana."""
    if ml_quantiles is not None:
        prior = GaussianPrior.from_quantiles(ml_quantiles, source="ml_xgboost")
    else:
        prior = GaussianPrior.uninformative()
        if verbose: print("[HYPATIA L2] Prior no informativo.")
        
    posterior = bayesian_update(prior, regression_result, n_samples)
    if verbose: print(posterior.summary())
    return posterior

def compare_posteriors_by_arc(full_series, ml_quantiles: dict, n_obs_list: list[int], true_dadt: Optional[float] = None) -> list[dict]:
    """Compara posterior para distintos arcos."""
    from .residuals import simulate_short_arc
    from .regression import estimate_ols_hac
    
    results, prior = [], GaussianPrior.from_quantiles(ml_quantiles)
    for n in n_obs_list:
        if n > full_series.n_points: continue
        short = simulate_short_arc(full_series, n)
        reg = estimate_ols_hac(short)
        post = bayesian_update(prior, reg)
        
        row = {"n_obs": n, "arc_years": float(short.times_years[-1]), "dadt_ols": reg.dadt_au_my, "dadt_post": post.mean}
        if true_dadt is not None:
            row["error_ols"] = abs(reg.dadt_au_my - true_dadt)
            row["error_post"] = abs(post.mean - true_dadt)
        results.append(row)
    return results