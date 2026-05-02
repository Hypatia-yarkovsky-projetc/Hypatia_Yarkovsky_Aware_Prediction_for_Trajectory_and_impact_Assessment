"""
visualizer.py
-------------
Visualizaciones de la Capa 2: residuos orbitales, descomposición STL,
estimaciones de da/dt y comparación de posteriors bayesianos.

Todas las figuras retornan objetos matplotlib Figure/Axes para que
puedan integrarse en el dashboard de Streamlit (Capa 3) o exportarse
directamente a archivos PNG/PDF para el poster científico.

Uso:
    from src.layer2_ts.visualizer import (
        plot_residuals,
        plot_regression_comparison,
        plot_sensitivity,
        plot_bayesian_update,
        plot_stl_decomposition,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from typing import Optional

from .residuals import ResidualSeries
from .regression import RegressionResult
from .bayesian import BayesianPosterior, GaussianPrior
from .diagnostics import DiagnosticsReport

# ── Paleta HYPATIA ────────────────────────────────────────────────────────
C_PURPLE  = "#534AB7"
C_TEAL    = "#1D9E75"
C_AMBER   = "#BA7517"
C_CORAL   = "#D85A30"
C_GRAY    = "#888780"
C_DARK    = "#1A1A1A"
C_BG      = "#FAFAF8"

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "grid.linewidth"    : 0.5,
    "figure.dpi"        : 130,
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
})


# ── 1. Residuos orbitales + tendencia ────────────────────────────────────

def plot_residuals(
    series          : ResidualSeries,
    regression      : Optional[RegressionResult] = None,
    true_dadt       : Optional[float] = None,
    title           : Optional[str] = None,
    save_path       : Optional[str] = None,
) -> plt.Figure:
    """
    Figura principal de la Capa 2: serie de residuos ε(t) con la
    tendencia ajustada superpuesta.

    Paneles:
        Superior : ε(t) en AU × 10⁻⁶ con banda de confianza
        Inferior : residuos del ajuste (para verificar homocedasticidad)

    Args:
        series     : ResidualSeries completa
        regression : RegressionResult opcional para superponer la tendencia
        true_dadt  : da/dt real (línea de referencia, si se conoce)
        title      : título de la figura
        save_path  : ruta para guardar la figura (None = no guardar)

    Returns:
        matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                              gridspec_kw={"height_ratios": [3, 1]},
                              facecolor=C_BG)
    fig.subplots_adjust(hspace=0.08)

    t   = series.times_years
    eps = series.epsilon * 1e6   # escalar a ×10⁻⁶ AU

    ax1, ax2 = axes
    ax1.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)

    # Residuos observados
    ax1.scatter(t, eps, s=18, color=C_GRAY, alpha=0.7,
                label="Residuos ε(t) observados", zorder=3)

    # Tendencia ajustada
    if regression is not None:
        t_fit  = regression.t_fit
        tr_fit = regression.trend_fit * 1e6
        ax1.plot(t_fit, tr_fit, color=C_PURPLE, lw=2.0,
                 label=f"Tendencia {regression.method.upper()} "
                       f"(da/dt={regression.dadt_au_my:+.3f} AU/My)", zorder=4)

        # Banda de confianza ±1.96σ alrededor de la tendencia
        sigma = regression.std_error * 1e6
        ax1.fill_between(
            t_fit,
            tr_fit - 1.96 * sigma * t_fit,
            tr_fit + 1.96 * sigma * t_fit,
            alpha=0.12, color=C_PURPLE, label="IC 95% de la pendiente"
        )

        # Residuos del ajuste en el panel inferior
        res_fit = regression.residuals_fit * 1e6
        ax2.scatter(t_fit, res_fit, s=10, color=C_GRAY, alpha=0.6)
        ax2.axhline(0, color=C_DARK, lw=0.8, ls="--")
        ax2.set_ylabel("Resid.\najuste\n(×10⁻⁶ AU)", fontsize=8)
        ax2.set_xlabel("Tiempo (años desde inicio del arco)", fontsize=10)

    # Línea de referencia da/dt real (si se conoce)
    if true_dadt is not None:
        t_line = np.linspace(t[0], t[-1], 100)
        y_line = true_dadt * 1e-6 * t_line * 1e6   # AU/My × años → ×10⁻⁶ AU
        ax1.plot(t_line, y_line, color=C_CORAL, lw=1.5, ls=":",
                 label=f"da/dt real = {true_dadt:+.3f} AU/My (JPL)", zorder=5)

    ax1.axhline(0, color=C_DARK, lw=0.6, ls="--", alpha=0.5)
    ax1.set_ylabel("Residuo ε(t) = a_obs − a_pred  (×10⁻⁶ AU)", fontsize=10)
    ax1.set_xticklabels([])
    ax1.legend(loc="upper left", framealpha=0.9)

    _title = title or f"Residuos orbitales — Asteroide {series.asteroid_id}  |  {series.n_points} obs"
    ax1.set_title(_title, fontsize=11, fontweight="bold", pad=10)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
        print(f"[HYPATIA L2] Figura guardada: {save_path}")

    return fig


# ── 2. Comparación de los tres métodos de regresión ───────────────────────

def plot_regression_comparison(
    series   : ResidualSeries,
    results  : dict[str, RegressionResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compara las estimaciones de da/dt de los tres métodos (OLS, HAC, STL)
    como un gráfico de coeficientes con intervalos de confianza.

    Args:
        series  : ResidualSeries
        results : dict {'ols': RegressionResult, 'ols_hac': ..., 'stl': ...}

    Returns:
        matplotlib.Figure
    """
    fig, (ax_coef, ax_fit) = plt.subplots(1, 2, figsize=(12, 5),
                                           facecolor=C_BG)
    ax_coef.set_facecolor(C_BG)
    ax_fit.set_facecolor(C_BG)

    colors = {"ols": C_TEAL, "ols_hac": C_PURPLE, "stl": C_AMBER}
    labels = {"ols": "OLS", "ols_hac": "OLS-HAC\n(Newey-West)", "stl": "STL"}
    methods = [m for m in ["ols", "ols_hac", "stl"] if m in results]

    # Panel izquierdo: forest plot de coeficientes
    y_pos = np.arange(len(methods))
    for i, method in enumerate(methods):
        r   = results[method]
        col = colors.get(method, C_GRAY)
        ax_coef.errorbar(
            r.dadt_au_my, i,
            xerr=[[r.dadt_au_my - r.ci_lower], [r.ci_upper - r.dadt_au_my]],
            fmt="o", color=col, ms=8, lw=2, capsize=5, capthick=2,
            label=f"{labels[method]}: {r.dadt_au_my:+.4f} ± {r.std_error:.4f}"
        )

    ax_coef.axvline(0, color=C_DARK, lw=0.8, ls="--", alpha=0.5)
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels([labels[m] for m in methods])
    ax_coef.set_xlabel("da/dt estimado (AU/My)")
    ax_coef.set_title("Estimaciones de da/dt con IC 95%", fontweight="bold")
    ax_coef.legend(loc="lower right", fontsize=8)

    # Panel derecho: tendencias ajustadas sobre los residuos
    t   = series.times_years
    eps = series.epsilon * 1e6

    ax_fit.scatter(t, eps, s=12, color=C_GRAY, alpha=0.5, zorder=2,
                   label="Residuos observados")

    for method in methods:
        r   = results[method]
        col = colors.get(method, C_GRAY)
        ax_fit.plot(r.t_fit, r.trend_fit * 1e6,
                    color=col, lw=2, label=labels[method], zorder=3)

    ax_fit.axhline(0, color=C_DARK, lw=0.6, ls="--", alpha=0.4)
    ax_fit.set_xlabel("Tiempo (años)")
    ax_fit.set_ylabel("ε(t) (×10⁻⁶ AU)")
    ax_fit.set_title("Tendencias ajustadas sobre residuos", fontweight="bold")
    ax_fit.legend(fontsize=8)

    fig.suptitle(
        f"Comparación de métodos — Asteroide {series.asteroid_id}",
        fontsize=12, fontweight="bold", y=1.01
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)

    return fig


# ── 3. Análisis de sensibilidad ───────────────────────────────────────────

def plot_sensitivity(
    sensitivity_df,
    true_dadt  : Optional[float] = None,
    save_path  : Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza cómo la estimación de da/dt y su incertidumbre varían
    con el número de observaciones disponibles (experimento de arco corto).

    Args:
        sensitivity_df : DataFrame de sensitivity_analysis()
        true_dadt      : da/dt real para comparar (línea de referencia)
        save_path      : ruta para guardar

    Returns:
        matplotlib.Figure
    """
    import pandas as pd

    df = sensitivity_df
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                    sharex=True, facecolor=C_BG)
    ax1.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)

    n = df["n_obs"].values
    dadt = df["dadt"].values
    ci_lo = df["ci_lower"].values
    ci_hi = df["ci_upper"].values
    ci_w  = df["ci_width"].values

    # Panel superior: estimación + IC
    ax1.fill_between(n, ci_lo, ci_hi, alpha=0.2, color=C_PURPLE,
                     label="IC 95%")
    ax1.plot(n, dadt, "o-", color=C_PURPLE, lw=2, ms=6,
             label="da/dt estimado")

    if true_dadt is not None:
        ax1.axhline(true_dadt, color=C_CORAL, lw=1.5, ls="--",
                    label=f"da/dt real = {true_dadt:+.3f} AU/My")

    ax1.set_ylabel("da/dt (AU/My)")
    ax1.set_title("Convergencia de la estimación de da/dt con el arco de observación",
                  fontweight="bold")
    ax1.legend()

    # Panel inferior: ancho del IC (incertidumbre)
    ax2.bar(n, ci_w, color=C_TEAL, alpha=0.7, label="Ancho IC 95%")
    ax2.set_xlabel("Número de observaciones")
    ax2.set_ylabel("Ancho IC 95% (AU/My)")
    ax2.set_title("Reducción de incertidumbre con más observaciones",
                  fontweight="bold")
    ax2.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)

    return fig


# ── 4. Actualización bayesiana (prior → posterior) ────────────────────────

def plot_bayesian_update(
    posterior  : BayesianPosterior,
    true_dadt  : Optional[float] = None,
    x_range    : Optional[tuple] = None,
    save_path  : Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza la actualización bayesiana: prior (ML) → likelihood (datos) → posterior.

    Args:
        posterior  : BayesianPosterior de bayesian_update()
        true_dadt  : da/dt real (línea vertical, si se conoce)
        x_range    : rango del eje x [AU/My] (None = automático)
        save_path  : ruta para guardar

    Returns:
        matplotlib.Figure
    """
    prior_dist = posterior.prior
    lik_dist   = posterior.likelihood

    # Rango del eje x
    all_means  = [prior_dist.mean, lik_dist.dadt_au_my, posterior.mean]
    all_stds   = [prior_dist.std, lik_dist.std_error, posterior.std]
    x_center   = np.mean(all_means)
    x_width    = max(3 * max(s for s in all_stds if s < 1e3), 0.3)
    if x_range is None:
        x_range = (x_center - x_width, x_center + x_width)

    x = np.linspace(x_range[0], x_range[1], 500)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    from scipy.stats import norm as scipy_norm

    # Prior (ML)
    if prior_dist.std < 1e3:
        y_prior = scipy_norm.pdf(x, prior_dist.mean, prior_dist.std)
        ax.fill_between(x, y_prior, alpha=0.18, color=C_AMBER)
        ax.plot(x, y_prior, color=C_AMBER, lw=2,
                label=f"Prior ML  μ={prior_dist.mean:+.3f}, σ={prior_dist.std:.3f}")

    # Likelihood (datos observacionales)
    if lik_dist.std_error < 1e3:
        y_lik = scipy_norm.pdf(x, lik_dist.dadt_au_my, lik_dist.std_error)
        ax.fill_between(x, y_lik, alpha=0.18, color=C_TEAL)
        ax.plot(x, y_lik, color=C_TEAL, lw=2,
                label=f"Likelihood ({lik_dist.method.upper()})  "
                      f"μ={lik_dist.dadt_au_my:+.3f}, σ={lik_dist.std_error:.3f}")

    # Posterior
    y_post = scipy_norm.pdf(x, posterior.mean, posterior.std)
    ax.fill_between(x, y_post, alpha=0.25, color=C_PURPLE)
    ax.plot(x, y_post, color=C_PURPLE, lw=2.5,
            label=f"Posterior  μ={posterior.mean:+.3f}, σ={posterior.std:.3f}")

    # IC 95% del posterior
    ax.axvline(posterior.ci_lower, color=C_PURPLE, lw=1, ls=":", alpha=0.7)
    ax.axvline(posterior.ci_upper, color=C_PURPLE, lw=1, ls=":",
               alpha=0.7, label=f"IC 95%: [{posterior.ci_lower:+.3f}, {posterior.ci_upper:+.3f}]")

    # da/dt real
    if true_dadt is not None:
        ax.axvline(true_dadt, color=C_CORAL, lw=2, ls="--",
                   label=f"Valor real JPL = {true_dadt:+.3f} AU/My")

    ax.set_xlabel("da/dt (AU/My)")
    ax.set_ylabel("Densidad de probabilidad")
    ax.set_title(
        f"Actualización Bayesiana  |  Peso prior: {posterior.weight_prior:.0%}  "
        f"· Peso datos: {posterior.weight_likelihood:.0%}",
        fontweight="bold"
    )
    ax.legend(loc="upper left", framealpha=0.9)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)

    return fig


# ── 5. Descomposición STL ─────────────────────────────────────────────────

def plot_stl_decomposition(
    series    : ResidualSeries,
    period    : Optional[int] = None,
    save_path : Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza la descomposición STL completa:
        ε(t) = Tendencia + Estacionalidad + Residuo

    Args:
        series  : ResidualSeries
        period  : período estacional en puntos (None = automático)
        save_path: ruta para guardar

    Returns:
        matplotlib.Figure
    """
    import pandas as pd
    from statsmodels.tsa.seasonal import STL

    n  = series.n_points
    t  = series.times_years
    eps = series.epsilon * 1e6

    if period is None:
        dt_med = np.median(np.diff(series.times_jd))
        period = max(2, int(round(365.25 / dt_med)))

    if n < 2 * period:
        fig, ax = plt.subplots(facecolor=C_BG)
        ax.text(0.5, 0.5,
                f"Serie muy corta para STL\n(n={n}, período={period})",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ts     = pd.Series(eps)
    result = STL(ts, period=period, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), facecolor=C_BG,
                              gridspec_kw={"hspace": 0.05})
    components = [
        (eps,                 "ε(t) original",    C_GRAY),
        (result.trend,        "Tendencia T(t)",   C_PURPLE),
        (result.seasonal,     "Estacionalidad S(t)", C_TEAL),
        (result.resid,        "Residuo R(t)",     C_AMBER),
    ]

    for ax, (data, label, color) in zip(axes, components):
        ax.set_facecolor(C_BG)
        ax.plot(t[:len(data)], data[:len(t)], color=color, lw=1.4)
        ax.axhline(0, color=C_DARK, lw=0.5, ls="--", alpha=0.4)
        ax.set_ylabel(label, fontsize=8)
        if ax is not axes[-1]:
            ax.set_xticklabels([])

    axes[0].set_title(
        f"Descomposición STL — Asteroide {series.asteroid_id}  |  período={period} obs",
        fontweight="bold", fontsize=11
    )
    axes[-1].set_xlabel("Tiempo (años)")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)

    return fig


# ── 6. Diagnóstico ACF/PACF ───────────────────────────────────────────────

def plot_acf_pacf(
    diag      : DiagnosticsReport,
    save_path : Optional[str] = None,
) -> plt.Figure:
    """
    Gráfico ACF y PACF de la serie de residuos.
    Útil para identificar si hay autocorrelación residual.

    Args:
        diag      : DiagnosticsReport de run_diagnostics()
        save_path : ruta para guardar

    Returns:
        matplotlib.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=C_BG)
    ax1.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)

    lags = diag.acf_lags
    n_approx = len(lags) * 3   # estimación del n original
    ci_bound = 1.96 / np.sqrt(n_approx)

    for ax, vals, title in [
        (ax1, diag.acf_values,  "ACF — Función de autocorrelación"),
        (ax2, diag.pacf_values, "PACF — Autocorrelación parcial"),
    ]:
        ax.bar(lags, vals, color=C_PURPLE, alpha=0.7, width=0.6)
        ax.axhline(0,         color=C_DARK, lw=0.8)
        ax.axhline(+ci_bound, color=C_CORAL, lw=1.2, ls="--",
                   label=f"IC 95% (±{ci_bound:.3f})")
        ax.axhline(-ci_bound, color=C_CORAL, lw=1.2, ls="--")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlación")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Autocorrelación — LB p={diag.lb_pvalue:.3f}  "
        f"({'autocorr. detectada' if diag.has_autocorr else 'sin autocorr.'})",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)

    return fig
