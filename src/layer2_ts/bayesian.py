"""
bayesian.py
-----------
Actualización bayesiana del parámetro da/dt.

Combina el prior del modelo ML (Capa 3) con la verosimilitud de los
datos observacionales (Capa 2) para producir una distribución posterior
de da/dt más informada que cualquiera de las dos fuentes por separado.

Marco teórico:
    Prior    : P(θ) provisto por el modelo ML de la Capa 3
               θ = da/dt ∼ N(μ_prior, σ_prior²)
               donde μ y σ se derivan de los cuantiles del modelo XGBoost

    Likelihood: P(datos|θ) derivada del estimador OLS/HAC/STL
               datos ∼ N(β₁·t, σ_noise²)  donde β₁ es función de θ

    Posterior : P(θ|datos) ∝ P(datos|θ) · P(θ)
               Para distribuciones gaussianas, la posterior es gaussiana
               con parámetros de actualización analíticos (conjugación exacta).

Fórmulas de actualización (prior gaussiano + likelihood gaussiana):
    σ_post² = (1/σ_prior² + 1/σ_likelihood²)⁻¹
    μ_post  = σ_post² · (μ_prior/σ_prior² + μ_likelihood/σ_likelihood²)

Cuando σ_prior → ∞ (prior no informativo): μ_post → μ_likelihood (OLS puro)
Cuando σ_likelihood → ∞ (pocos datos):     μ_post → μ_prior (ML domina)

Integración con las capas:
    Capa 3 → prior: dict con cuantiles {0.10, 0.25, 0.50, 0.75, 0.90}
    Capa 2 → likelihood: RegressionResult de estimate_ols_hac() o estimate_stl()
    Salida → BayesianPosterior: distribucion completa + muestras MCMC
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm, truncnorm

from .regression import RegressionResult


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class GaussianPrior:
    """
    Prior gaussiano sobre da/dt derivado de los cuantiles del modelo ML.

    Se construye desde los cuantiles {P10, P50, P90} asumiendo normalidad:
        μ_prior = P50
        σ_prior = (P90 − P10) / (2 × 1.2816)  [factor de z-score para 80% IC]
    """
    mean        : float   # μ_prior en AU/My
    std         : float   # σ_prior en AU/My
    source      : str = "ml_quantile"

    @classmethod
    def from_quantiles(cls, quantiles: dict, source: str = "ml_quantile") -> "GaussianPrior":
        """
        Construye el prior desde un dict de cuantiles del modelo ML.

        Args:
            quantiles : dict {0.10: val, 0.25: val, 0.50: val, 0.75: val, 0.90: val}
                        Unidades: AU/My
            source    : descripción del origen del prior

        Returns:
            GaussianPrior
        """
        mu = float(quantiles[0.50])

        # Estimar σ desde el IC 80% (P10–P90)
        # P90 − P10 = 2 × 1.2816 × σ  (para distribución normal)
        if 0.10 in quantiles and 0.90 in quantiles:
            sigma = (float(quantiles[0.90]) - float(quantiles[0.10])) / (2 * 1.2816)
        elif 0.25 in quantiles and 0.75 in quantiles:
            # Desde IQR: P75 − P25 = 2 × 0.6745 × σ
            sigma = (float(quantiles[0.75]) - float(quantiles[0.25])) / (2 * 0.6745)
        else:
            sigma = abs(mu) * 0.5   # fallback: 50% de incertidumbre relativa

        sigma = max(sigma, 1e-6)   # protección numérica
        return cls(mean=mu, std=sigma, source=source)

    @classmethod
    def uninformative(cls) -> "GaussianPrior":
        """Prior no informativo: equivalente a confiar solo en los datos."""
        return cls(mean=0.0, std=1e6, source="uninformative")

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return norm.pdf(x, self.mean, self.std)

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(self.mean, self.std, n)

    def summary(self) -> str:
        return (f"Prior [{self.source}]: "
                f"μ={self.mean:+.4f}, σ={self.std:.4f}, "
                f"IC80%=[{self.mean-1.2816*self.std:+.4f}, "
                f"{self.mean+1.2816*self.std:+.4f}] AU/My")


@dataclass
class BayesianPosterior:
    """
    Distribución posterior de da/dt tras la actualización bayesiana.

    Contiene tanto los parámetros analíticos (gaussiana conjugada)
    como las muestras MCMC para estimaciones no gaussianas.
    """
    mean        : float   # μ_post en AU/My
    std         : float   # σ_post en AU/My
    ci_lower    : float   # percentil 2.5% del posterior
    ci_upper    : float   # percentil 97.5% del posterior

    prior       : GaussianPrior
    likelihood  : RegressionResult

    # Pesos relativos (cuánto aporta cada fuente)
    weight_prior      : float   # 0-1, mayor → prior más informativo que datos
    weight_likelihood : float   # 1 - weight_prior

    # Muestras Monte Carlo para propagación de incertidumbre
    samples     : np.ndarray = field(default_factory=lambda: np.array([]))
    n_samples   : int = 0

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def sample_dadt(self, n: int = 1000, seed: int = 42) -> np.ndarray:
        """Genera muestras de da/dt desde la posterior gaussiana."""
        rng = np.random.default_rng(seed)
        return rng.normal(self.mean, self.std, n)

    def to_layer1_input(self) -> dict:
        """
        Convierte la posterior al formato que espera la Capa 1.
        Úsalo como argumento de propagate_from_state().

        Returns:
            dict {'dadt_mean': float, 'dadt_std': float, 'samples': ndarray}
        """
        return {
            "dadt_mean": self.mean,
            "dadt_std" : self.std,
            "ci_lower" : self.ci_lower,
            "ci_upper" : self.ci_upper,
            "samples"  : self.sample_dadt(200),
        }

    def summary(self) -> str:
        dom = "prior" if self.weight_prior > 0.5 else "datos"
        lines = [
            "─" * 52,
            "  POSTERIOR BAYESIANA — da/dt",
            "─" * 52,
            f"  μ posterior   : {self.mean:+.4f} AU/My",
            f"  σ posterior   : {self.std:.4f} AU/My",
            f"  IC 95%        : [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}] AU/My",
            f"  Peso prior    : {self.weight_prior:.1%}  ({self.prior.source})",
            f"  Peso datos    : {self.weight_likelihood:.1%}  ({self.likelihood.method})",
            f"  Dominante     : {dom}",
            "─" * 52,
            f"  {self.prior.summary()}",
            f"  Likelihood: μ={self.likelihood.dadt_au_my:+.4f}, "
              f"σ={self.likelihood.std_error:.4f} AU/My",
            "─" * 52,
        ]
        return "\n".join(lines)


# ── Actualización bayesiana analítica ─────────────────────────────────────

def bayesian_update(
    prior      : GaussianPrior,
    likelihood : RegressionResult,
    n_samples  : int = 2000,
    seed       : int = 42,
    alpha      : float = 0.05,
) -> BayesianPosterior:
    """
    Actualización bayesiana con prior y likelihood gaussianos (conjugada exacta).

    Fórmulas:
        τ_prior = 1 / σ_prior²        (precisión del prior)
        τ_data  = 1 / σ_likelihood²   (precisión de los datos)
        τ_post  = τ_prior + τ_data
        μ_post  = (τ_prior·μ_prior + τ_data·μ_data) / τ_post
        σ_post  = 1 / √τ_post

    Args:
        prior      : GaussianPrior de la Capa 3
        likelihood : RegressionResult de estimate_ols_hac() o estimate_stl()
        n_samples  : muestras Monte Carlo del posterior
        seed       : semilla para reproducibilidad
        alpha      : nivel para el IC (default 0.05 → IC 95%)

    Returns:
        BayesianPosterior con todos los parámetros calculados
    """
    # Precisiones
    tau_prior = 1.0 / (prior.std ** 2)
    tau_data  = 1.0 / (likelihood.std_error ** 2) if likelihood.std_error > 1e-10 else 0.0

    tau_post  = tau_prior + tau_data

    # Parámetros del posterior
    mu_post  = (tau_prior * prior.mean + tau_data * likelihood.dadt_au_my) / tau_post
    std_post = 1.0 / np.sqrt(tau_post)

    # Intervalo de credibilidad
    z_crit  = norm.ppf(1 - alpha / 2)
    ci_lo   = mu_post - z_crit * std_post
    ci_hi   = mu_post + z_crit * std_post

    # Pesos relativos (fracción de la precisión total)
    w_prior = tau_prior / tau_post
    w_data  = tau_data  / tau_post

    # Muestras Monte Carlo
    rng     = np.random.default_rng(seed)
    samples = rng.normal(mu_post, std_post, n_samples)

    return BayesianPosterior(
        mean              = float(mu_post),
        std               = float(std_post),
        ci_lower          = float(ci_lo),
        ci_upper          = float(ci_hi),
        prior             = prior,
        likelihood        = likelihood,
        weight_prior      = float(w_prior),
        weight_likelihood = float(w_data),
        samples           = samples,
        n_samples         = n_samples,
    )


# ── Función de alto nivel ─────────────────────────────────────────────────

def full_bayesian_estimation(
    regression_result : RegressionResult,
    ml_quantiles      : Optional[dict] = None,
    dadt_prior_mean   : Optional[float] = None,
    dadt_prior_std    : Optional[float] = None,
    n_samples         : int = 2000,
    verbose           : bool = True,
) -> BayesianPosterior:
    """
    Función de alto nivel para la estimación bayesiana completa.

    Acepta el prior en dos formatos:
        1. Dict de cuantiles del modelo ML (formato directo de la Capa 3)
        2. Media y desviación estándar explícitas

    Args:
        regression_result : RegressionResult de cualquier método de regresión
        ml_quantiles      : dict {0.10: val, ..., 0.90: val} de la Capa 3
        dadt_prior_mean   : media del prior [AU/My] (si no hay cuantiles ML)
        dadt_prior_std    : desviación estándar del prior [AU/My]
        n_samples         : muestras Monte Carlo
        verbose           : imprimir resumen

    Returns:
        BayesianPosterior
    """
    # Construir el prior
    if ml_quantiles is not None:
        prior = GaussianPrior.from_quantiles(ml_quantiles, source="ml_xgboost")
    elif dadt_prior_mean is not None and dadt_prior_std is not None:
        prior = GaussianPrior(
            mean=dadt_prior_mean, std=dadt_prior_std, source="manual"
        )
    else:
        prior = GaussianPrior.uninformative()
        if verbose:
            print("[HYPATIA L2] Prior no informativo: usando solo datos observacionales")

    # Actualización bayesiana
    posterior = bayesian_update(prior, regression_result, n_samples=n_samples)

    if verbose:
        print(posterior.summary())

    return posterior


def compare_posteriors_by_arc(
    full_series   : "ResidualSeries",
    ml_quantiles  : dict,
    n_obs_list    : list[int] = [5, 10, 20, 30],
    true_dadt     : Optional[float] = None,
) -> list[dict]:
    """
    Compara la posterior bayesiana para distintos arcos de observación.

    Demuestra la contribución del prior ML: con pocos datos, la posterior
    se acerca al prior; con muchos datos, converge a la estimación OLS.

    Args:
        full_series  : ResidualSeries completa
        ml_quantiles : cuantiles del modelo ML (Capa 3)
        n_obs_list   : lista de arcos a probar
        true_dadt    : da/dt real (para calcular error si se conoce)

    Returns:
        Lista de dicts con los resultados para cada arco
    """
    from .residuals import simulate_short_arc
    from .regression import estimate_ols_hac

    results = []
    prior = GaussianPrior.from_quantiles(ml_quantiles)

    for n in n_obs_list:
        if n > full_series.n_points:
            continue
        short = simulate_short_arc(full_series, n)
        reg   = estimate_ols_hac(short)
        post  = bayesian_update(prior, reg)

        row = {
            "n_obs"        : n,
            "arc_years"    : float(short.times_years[-1]),
            "dadt_ols"     : reg.dadt_au_my,
            "dadt_post"    : post.mean,
            "ci_width_ols" : reg.ci_width,
            "ci_width_post": post.ci_width,
            "weight_prior" : post.weight_prior,
            "weight_data"  : post.weight_likelihood,
        }
        if true_dadt is not None:
            row["error_ols"]  = abs(reg.dadt_au_my - true_dadt)
            row["error_post"] = abs(post.mean - true_dadt)

        results.append(row)

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print(f"\n[HYPATIA L2] Comparación posterior por arco:")
        print(df.to_string(index=False, float_format="{:+.4f}".format))

    return results
