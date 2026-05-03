"""
visualizer.py
Visualizaciones de la Capa 2: residuos, STL, comparacion de metodos y actualizacion bayesiana.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.gridspec import GridSpec

from .residuals import ResidualSeries
from .regression import RegressionResult
from .bayesian import BayesianPosterior, GaussianPrior
from .diagnostics import DiagnosticsReport

# Paleta oficial
C_PURPLE, C_TEAL, C_AMBER, C_CORAL, C_GRAY, C_DARK, C_BG = "#534AB7", "#1D9E75", "#BA7517", "#D85A30", "#888780", "#1A1A1A", "#FAFAF8"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})

def plot_residuals(series: ResidualSeries, regression: Optional[RegressionResult] = None,
                   true_dadt: Optional[float] = None, save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, facecolor=C_BG)
    fig.subplots_adjust(hspace=0.08)
    t, eps = series.times_years, series.epsilon * 1e6
    ax1, ax2 = axes
    ax1.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)

    ax1.scatter(t, eps, s=18, color=C_GRAY, alpha=0.7, label="Residuos observados", zorder=3)
    if regression is not None:
        ax1.plot(regression.t_fit, regression.trend_fit * 1e6, color=C_PURPLE, lw=2,
                 label=f"Tendencia {regression.method.upper()}", zorder=4)
        sigma = regression.std_error * 1e6
        ax1.fill_between(regression.t_fit, regression.trend_fit * 1e6 - 1.96 * sigma * regression.t_fit,
                         regression.trend_fit * 1e6 + 1.96 * sigma * regression.t_fit,
                         alpha=0.12, color=C_PURPLE, label="IC 95% pendiente")
        ax2.scatter(regression.t_fit, regression.residuals_fit * 1e6, s=10, color=C_GRAY, alpha=0.6)
        ax2.axhline(0, color=C_DARK, lw=0.8, ls="--")
        ax2.set_ylabel("Residuos ajuste (x10^-6 AU)", fontsize=8)

    if true_dadt is not None:
        t_line = np.linspace(t[0], t[-1], 100)
        ax1.plot(t_line, true_dadt * t_line, color=C_CORAL, lw=1.5, ls=":", label=f"da/dt real = {true_dadt:+.3f}")

    ax1.axhline(0, color=C_DARK, lw=0.6, ls="--", alpha=0.5)
    ax1.set_ylabel("Residuo epsilon(t) (x10^-6 AU)")
    ax1.set_xticklabels([])
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_title(f"Residuos orbitales - Asteroide {series.asteroid_id}", fontsize=11, fontweight="bold", pad=10)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_regression_comparison(series: ResidualSeries, results: dict[str, RegressionResult],
                               save_path: Optional[str] = None) -> plt.Figure:
    fig, (ax_coef, ax_fit) = plt.subplots(1, 2, figsize=(12, 5), facecolor=C_BG)
    ax_coef.set_facecolor(C_BG)
    ax_fit.set_facecolor(C_BG)
    colors = {"ols": C_TEAL, "ols_hac": C_PURPLE, "stl": C_AMBER}
    labels = {"ols": "OLS", "ols_hac": "OLS-HAC", "stl": "STL"}
    methods = [m for m in ["ols", "ols_hac", "stl"] if m in results]

    y_pos = np.arange(len(methods))
    for i, method in enumerate(methods):
        r, col = results[method], colors.get(method, C_GRAY)
        ax_coef.errorbar(r.dadt_au_my, i, xerr=[[r.dadt_au_my - r.ci_lower], [r.ci_upper - r.dadt_au_my]],
                         fmt="o", color=col, ms=8, lw=2, capsize=5, label=f"{labels[method]}: {r.dadt_au_my:+.4f}")
    ax_coef.axvline(0, color=C_DARK, lw=0.8, ls="--")
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels([labels[m] for m in methods])
    ax_coef.set_xlabel("da/dt estimado (AU/My)")
    ax_coef.set_title("Estimaciones con IC 95%", fontweight="bold")
    ax_coef.legend(loc="lower right", fontsize=8)

    t, eps = series.times_years, series.epsilon * 1e6
    ax_fit.scatter(t, eps, s=12, color=C_GRAY, alpha=0.5, label="Residuos", zorder=2)
    for method in methods:
        r, col = results[method], colors.get(method, C_GRAY)
        ax_fit.plot(r.t_fit, r.trend_fit * 1e6, color=col, lw=2, label=labels[method], zorder=3)
    ax_fit.axhline(0, color=C_DARK, lw=0.6, ls="--")
    ax_fit.set_xlabel("Tiempo (anos)")
    ax_fit.set_ylabel("epsilon(t) (x10^-6 AU)")
    ax_fit.set_title("Tendencias ajustadas", fontweight="bold")
    ax_fit.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_sensitivity(sensitivity_df, true_dadt: Optional[float] = None, save_path: Optional[str] = None) -> plt.Figure:
    import pandas as pd
    df = sensitivity_df
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=C_BG)
    ax1.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)
    n, dadt, ci_lo, ci_hi, ci_w = df["n_obs"].values, df["dadt"].values, df["ci_lower"].values, df["ci_upper"].values, df["ci_width"].values

    ax1.fill_between(n, ci_lo, ci_hi, alpha=0.2, color=C_PURPLE, label="IC 95%")
    ax1.plot(n, dadt, "o-", color=C_PURPLE, lw=2, ms=6, label="da/dt estimado")
    if true_dadt is not None:
        ax1.axhline(true_dadt, color=C_CORAL, lw=1.5, ls="--", label=f"Real = {true_dadt:+.3f}")
    ax1.set_ylabel("da/dt (AU/My)")
    ax1.set_title("Convergencia de estimacion con arco", fontweight="bold")
    ax1.legend()

    ax2.bar(n, ci_w, color=C_TEAL, alpha=0.7, label="Ancho IC 95%")
    ax2.set_xlabel("Numero de observaciones")
    ax2.set_ylabel("Ancho IC 95%")
    ax2.set_title("Reduccion de incertidumbre", fontweight="bold")
    ax2.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_bayesian_update(posterior: BayesianPosterior, true_dadt: Optional[float] = None,
                         x_range: Optional[tuple] = None, save_path: Optional[str] = None) -> plt.Figure:
    from scipy.stats import norm as scipy_norm
    prior_dist, lik_dist = posterior.prior, posterior.likelihood
    all_means = [prior_dist.mean, lik_dist.dadt_au_my, posterior.mean]
    all_stds = [prior_dist.std, lik_dist.std_error, posterior.std]
    x_center, x_width = np.mean(all_means), max(3 * max(s for s in all_stds if s < 1e3), 0.3)
    if x_range is None:
        x_range = (x_center - x_width, x_center + x_width)
    x = np.linspace(x_range[0], x_range[1], 500)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    if prior_dist.std < 1e3:
        y_prior = scipy_norm.pdf(x, prior_dist.mean, prior_dist.std)
        ax.fill_between(x, y_prior, alpha=0.18, color=C_AMBER)
        ax.plot(x, y_prior, color=C_AMBER, lw=2, label=f"Prior ML: {prior_dist.mean:+.3f}")

    if lik_dist.std_error < 1e3:
        y_lik = scipy_norm.pdf(x, lik_dist.dadt_au_my, lik_dist.std_error)
        ax.fill_between(x, y_lik, alpha=0.18, color=C_TEAL)
        ax.plot(x, y_lik, color=C_TEAL, lw=2, label=f"Likelihood: {lik_dist.dadt_au_my:+.3f}")

    y_post = scipy_norm.pdf(x, posterior.mean, posterior.std)
    ax.fill_between(x, y_post, alpha=0.25, color=C_PURPLE)
    ax.plot(x, y_post, color=C_PURPLE, lw=2.5, label=f"Posterior: {posterior.mean:+.3f}")

    ax.axvline(posterior.ci_lower, color=C_PURPLE, lw=1, ls=":", alpha=0.7)
    ax.axvline(posterior.ci_upper, color=C_PURPLE, lw=1, ls=":", alpha=0.7, label=f"IC 95%: [{posterior.ci_lower:+.3f}, {posterior.ci_upper:+.3f}]")
    if true_dadt is not None:
        ax.axvline(true_dadt, color=C_CORAL, lw=2, ls="--", label=f"Real = {true_dadt:+.3f}")

    ax.set_xlabel("da/dt (AU/My)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Actualizacion Bayesiana | Peso prior: {posterior.weight_prior:.0%}", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_stl_decomposition(series: ResidualSeries, period: Optional[int] = None, save_path: Optional[str] = None) -> plt.Figure:
    import pandas as pd
    from statsmodels.tsa.seasonal import STL
    n, t, eps = series.n_points, series.times_years, series.epsilon * 1e6
    if period is None:
        dt_med = np.median(np.diff(series.times_jd))
        period = max(2, int(round(365.25 / dt_med)))
    if n < 2 * period:
        fig, ax = plt.subplots(facecolor=C_BG)
        ax.text(0.5, 0.5, f"Serie corta para STL (n={n})", ha="center", va="center", transform=ax.transAxes)
        return fig

    ts = pd.Series(eps, index=pd.RangeIndex(n))
    result = STL(ts, period=period, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), facecolor=C_BG, gridspec_kw={"hspace": 0.05})
    components = [(eps, "epsilon(t)", C_GRAY), (result.trend, "Tendencia", C_PURPLE),
                  (result.seasonal, "Estacionalidad", C_TEAL), (result.resid, "Residuo", C_AMBER)]
    for ax, (data, label, color) in zip(axes, components):
        ax.set_facecolor(C_BG)
        ax.plot(t[:len(data)], data[:len(t)], color=color, lw=1.4)
        ax.axhline(0, color=C_DARK, lw=0.5, ls="--", alpha=0.4)
        ax.set_ylabel(label, fontsize=8)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[0].set_title(f"Descomposicion STL - Asteroide {series.asteroid_id}", fontweight="bold")
    axes[-1].set_xlabel("Tiempo (anos)")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_acf_pacf(diag: DiagnosticsReport, save_path: Optional[str] = None) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=C_BG)
    ax1.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)
    lags, n_approx = diag.acf_lags, len(diag.acf_lags) * 3
    ci_bound = 1.96 / np.sqrt(n_approx)
    for ax, vals, title in [(ax1, diag.acf_values, "ACF"), (ax2, diag.pacf_values, "PACF")]:
        ax.bar(lags, vals, color=C_PURPLE, alpha=0.7, width=0.6)
        ax.axhline(0, color=C_DARK, lw=0.8)
        ax.axhline(+ci_bound, color=C_CORAL, lw=1.2, ls="--", label=f"IC 95% (+/-{ci_bound:.3f})")
        ax.axhline(-ci_bound, color=C_CORAL, lw=1.2, ls="--")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlacion")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle(f"Autocorrelacion - LB p={diag.lb_pvalue:.3f}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig