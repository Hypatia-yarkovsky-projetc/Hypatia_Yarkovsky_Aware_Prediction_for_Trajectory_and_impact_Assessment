"""
visualizer.py
Visualizaciones Capa 3: distribución de da/dt, importancia de features,
validación LOO-CV y calibración cuantílica.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from .model import HypatiaModel, ValidationReport, QUANTILES
from .features import FEATURE_NAMES, FEATURE_DESCRIPTIONS

C_PURPLE, C_TEAL, C_AMBER, C_CORAL, C_GRAY, C_BG = "#534AB7", "#1D9E75", "#BA7517", "#D85A30", "#888780", "#FAFAF8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5, "figure.dpi": 130,
})

def plot_prediction_distribution(inference_result, true_dadt=None, title=None, save_path=None) -> plt.Figure:
    """Distribución predicha P(da/dt|features) con cuantiles."""
    from scipy.interpolate import interp1d
    q_vals = inference_result.quantiles
    qs = sorted(q_vals.keys())
    median = q_vals[0.50]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    spread = inference_result.ci_80_upper - inference_result.ci_80_lower
    x_min, x_max = median - spread * 2, median + spread * 2
    x_lin = np.linspace(x_min, x_max, 400)

    try:
        inv_cdf = interp1d(qs, [q_vals[q] for q in qs], kind="linear", fill_value="extrapolate")
        fine_qs = np.linspace(0.05, 0.95, 200)
        fine_x = inv_cdf(fine_qs)
        pdf_y = np.maximum(np.gradient(fine_qs, fine_x), 0)
        ax.fill_between(fine_x, pdf_y, alpha=0.25, color=C_AMBER)
        ax.plot(fine_x, pdf_y, color=C_AMBER, lw=1.5, alpha=0.7)
    except Exception:
        pass

    colors_q = {0.10: C_TEAL, 0.25: C_PURPLE, 0.50: C_CORAL, 0.75: C_PURPLE, 0.90: C_TEAL}
    labels_q = {0.10: "Q10", 0.25: "Q25", 0.50: "Q50 (mediana)", 0.75: "Q75", 0.90: "Q90"}
    for q in qs:
        lw = 2.5 if q == 0.50 else 1.5
        ls = "-" if q == 0.50 else "--"
        ax.axvline(q_vals[q], color=colors_q[q], lw=lw, ls=ls, label=f"{labels_q[q]}: {q_vals[q]:+.4f} AU/My")
    ax.axvspan(q_vals[0.10], q_vals[0.90], alpha=0.08, color=C_TEAL, label="IC 80%")
    if true_dadt is not None:
        ax.axvline(true_dadt, color=C_CORAL, lw=2, ls=":", label=f"Real (JPL) = {true_dadt:+.4f} AU/My")

    ax.set_xlabel("da/dt (AU/My)")
    ax.set_ylabel("Densidad (relativa)")
    ax.set_title(title or "Distribucion predicha de da/dt — HYPATIA Capa 3", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left" if median < 0 else "upper right")
    if save_path: fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_feature_importance(model: HypatiaModel, save_path=None) -> plt.Figure:
    """Importancia de features del modelo Q50."""
    if not model.is_fitted or 0.50 not in model.quantile_models:
        raise RuntimeError("Modelo no entrenado.")
    importances = model.quantile_models[0.50].feature_importances_
    feat_names = model.feature_names
    descriptions = [FEATURE_DESCRIPTIONS.get(f, f) for f in feat_names]
    idx_sorted = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    colors = [C_PURPLE if i == idx_sorted[0] else C_TEAL for i in range(len(importances))]
    bars = ax.barh([descriptions[i] for i in idx_sorted], [importances[i] for i in idx_sorted],
                   color=[colors[i] for i in idx_sorted], alpha=0.8, edgecolor="none")
    for bar, val in zip(bars, [importances[i] for i in idx_sorted]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importancia (ganancia media)")
    ax.set_title("Feature Importance — XGBoost (Q50)", fontweight="bold")
    if save_path: fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_loocv_scatter(report: ValidationReport, true_dadt=None, save_path=None) -> plt.Figure:
    """Scatter predicho vs real (LOO-CV) y distribución de errores."""
    y_true, y_pred = report.y_true, report.y_pred_loocv
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=C_BG)
    ax1.set_facecolor(C_BG); ax2.set_facecolor(C_BG)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.1
    ax1.scatter(y_true, y_pred, s=20, alpha=0.6, color=C_PURPLE, label=f"n={len(y_true)}")
    ax1.plot([-lim, lim], [-lim, lim], color=C_CORAL, lw=1.5, ls="--", label="Prediccion perfecta (y=x)")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("da/dt real (AU/My)")
    ax1.set_ylabel("da/dt predicho LOO-CV (AU/My)")
    ax1.set_title(f"LOO-CV: predicho vs real\nRMSE={report.rmse_loocv:.4f}  R2={report.r2_loocv:.3f}", fontweight="bold")
    ax1.legend(fontsize=9)
    errors = y_pred - y_true
    ax2.hist(errors, bins=20, color=C_TEAL, alpha=0.7, edgecolor="white")
    ax2.axvline(0, color=C_CORAL, lw=2, ls="--", label="Error = 0")
    ax2.axvline(errors.mean(), color=C_AMBER, lw=1.5, ls=":", label=f"Media = {errors.mean():.4f}")
    ax2.set_xlabel("Error = predicho - real (AU/My)")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribucion de errores LOO-CV", fontweight="bold")
    ax2.legend(fontsize=9)
    fig.suptitle("Validacion Leave-One-Out Cross Validation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_dataset_distribution(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Distribución del dataset: da/dt, diámetro, taxonomía."""
    fig = plt.figure(figsize=(14, 5), facecolor=C_BG)
    axes = fig.subplots(1, 3)
    for ax in axes: ax.set_facecolor(C_BG)
    axes[0].hist(df["dadt_AuMy"], bins=25, color=C_PURPLE, alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color=C_CORAL, lw=2, ls="--")
    axes[0].axvline(df["dadt_AuMy"].median(), color=C_AMBER, lw=1.5, ls=":", label=f"Mediana={df['dadt_AuMy'].median():.3f}")
    axes[0].set_xlabel("da/dt (AU/My)"); axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribucion de da/dt", fontweight="bold"); axes[0].legend(fontsize=8)
    diam = df["diameter_km"].clip(0.01)
    dadt_abs = df["dadt_AuMy"].abs().clip(1e-5)
    axes[1].scatter(diam, dadt_abs, s=15, alpha=0.5, color=C_TEAL)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Diámetro (km)"); axes[1].set_ylabel("|da/dt| (AU/My)")
    axes[1].set_title("da/dt vs Diámetro (escala log)", fontweight="bold")
    d_line = np.logspace(np.log10(diam.min()), np.log10(diam.max()), 50)
    axes[1].plot(d_line, dadt_abs.median() * diam.median() / d_line, color=C_CORAL, lw=1.5, ls="--", label="∝ 1/D")
    axes[1].legend(fontsize=8)
    tax_counts = df["taxonomy"].str[0].str.upper().value_counts().head(8)
    axes[2].bar(tax_counts.index, tax_counts.values, color=C_AMBER, alpha=0.8, edgecolor="white")
    axes[2].set_xlabel("Clase taxonomica"); axes[2].set_ylabel("Numero de asteroides")
    axes[2].set_title("Distribucion taxonomica", fontweight="bold")
    fig.suptitle(f"Dataset de entrenamiento HYPATIA L3 — {len(df)} asteroides", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig

def plot_quantile_calibration(report: ValidationReport, df: pd.DataFrame, model: HypatiaModel, save_path=None) -> plt.Figure:
    """Calibración cuantílica: frecuencia empírica vs nominal."""
    from .features import extract_features
    from .dataset import TARGET_NAME
    X = extract_features(df)
    y = df[TARGET_NAME].values
    empirical = []
    for q in QUANTILES:
        q_pred = model.quantile_models[q].predict(X.values)
        empirical.append(np.mean(y <= q_pred))
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.plot([0, 1], [0, 1], color=C_CORAL, lw=2, ls="--", label="Calibracion perfecta")
    ax.scatter(QUANTILES, empirical, s=80, color=C_PURPLE, zorder=5, label="Calibracion del modelo")
    ax.plot(QUANTILES, empirical, color=C_PURPLE, lw=1.5, alpha=0.6)
    for q, emp in zip(QUANTILES, empirical):
        ax.annotate(f"Q{int(q*100)}\n({emp:.2f})", (q, emp), textcoords="offset points", xytext=(10, -10), fontsize=8, color=C_GRAY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Cuantil nominal"); ax.set_ylabel("Frecuencia empirica")
    ax.set_title("Calibracion cuantilica", fontweight="bold"); ax.legend(fontsize=9)
    if save_path: fig.savefig(save_path, bbox_inches="tight", facecolor=C_BG)
    return fig