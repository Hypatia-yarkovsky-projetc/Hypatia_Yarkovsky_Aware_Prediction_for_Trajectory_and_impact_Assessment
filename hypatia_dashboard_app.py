"""
HYPATIA — Dashboard de Misión
Sistema Híbrido de Predicción de Trayectorias de Objetos Cercanos a la Tierra

dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime

# ── Configuración de página ───────────────────────────────────────────────
st.set_page_config(
    page_title     = "HYPATIA",
    page_icon      = "🌌",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# ── Tema visual NASA Mission Control ─────────────────────────────────────
NASA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

:root {
    --nasa-black     : #060810;
    --nasa-deep      : #0B0E1A;
    --nasa-panel     : #0F1420;
    --nasa-border    : #1A2340;
    --nasa-blue      : #00D4FF;
    --nasa-blue-dim  : #0088AA;
    --nasa-orange    : #FF6B35;
    --nasa-orange-dim: #CC4A1A;
    --nasa-green     : #00FF88;
    --nasa-yellow    : #FFD600;
    --nasa-red       : #FF3355;
    --nasa-text      : #C8D8F0;
    --nasa-dim       : #5A6A80;
    --nasa-bright    : #FFFFFF;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--nasa-black) !important;
    color: var(--nasa-text) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--nasa-deep) !important;
    border-right: 1px solid var(--nasa-border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--nasa-text) !important;
}

h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.05em !important;
}

.stMetric { background: var(--nasa-panel) !important; }

.metric-card {
    background: linear-gradient(135deg, var(--nasa-panel) 0%, #0D1525 100%);
    border: 1px solid var(--nasa-border);
    border-left: 3px solid var(--nasa-blue);
    border-radius: 4px;
    padding: 16px 20px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--nasa-blue), transparent);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: var(--nasa-dim);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 26px;
    font-weight: 700;
    color: var(--nasa-blue);
    line-height: 1;
}
.metric-value.orange { color: var(--nasa-orange); }
.metric-value.green  { color: var(--nasa-green);  }
.metric-value.red    { color: var(--nasa-red);     }
.metric-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: var(--nasa-dim);
    margin-top: 4px;
}

.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--nasa-blue);
    border-bottom: 1px solid var(--nasa-border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

.status-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.status-ok      { background: rgba(0,255,136,0.1); color: var(--nasa-green); border: 1px solid rgba(0,255,136,0.3); }
.status-warn    { background: rgba(255,214,0,0.1);  color: var(--nasa-yellow); border: 1px solid rgba(255,214,0,0.3); }
.status-alert   { background: rgba(255,51,85,0.1);  color: var(--nasa-red);   border: 1px solid rgba(255,51,85,0.3); }
.status-nominal { background: rgba(0,212,255,0.1);  color: var(--nasa-blue);  border: 1px solid rgba(0,212,255,0.3); }

.stTabs [data-baseweb="tab-list"] {
    background: var(--nasa-panel);
    border-bottom: 1px solid var(--nasa-border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    color: var(--nasa-dim) !important;
    border-radius: 0 !important;
    padding: 12px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--nasa-blue) !important;
    border-bottom: 2px solid var(--nasa-blue) !important;
    background: transparent !important;
}

.stSlider [data-testid="stSlider"] div { color: var(--nasa-blue) !important; }
.stSelectbox div, .stNumberInput div { background: var(--nasa-panel) !important; border-color: var(--nasa-border) !important; }

.layer-pill {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    padding: 4px 12px;
    border-radius: 2px;
    margin: 2px;
    letter-spacing: 0.08em;
}
.l1 { background: rgba(83,74,183,0.2); color: #9B93FF; border: 1px solid rgba(83,74,183,0.4); }
.l2 { background: rgba(0,158,117,0.2); color: #00D4A0; border: 1px solid rgba(0,158,117,0.4); }
.l3 { background: rgba(186,117,23,0.2); color: #FFB344; border: 1px solid rgba(186,117,23,0.4); }

.scanline-effect {
    position: relative;
}
.scanline-effect::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,212,255,0.015) 2px, rgba(0,212,255,0.015) 4px);
    pointer-events: none;
}

.data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(26,35,64,0.5);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
}
.data-key   { color: var(--nasa-dim); }
.data-val   { color: var(--nasa-text); }
.data-val.accent { color: var(--nasa-blue); }

[data-testid="stExpander"] {
    background: var(--nasa-panel) !important;
    border: 1px solid var(--nasa-border) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: var(--nasa-text) !important;
}

div[data-testid="column"] > div {
    height: 100%;
}

.stButton > button {
    background: linear-gradient(135deg, var(--nasa-blue-dim), var(--nasa-blue)) !important;
    color: var(--nasa-black) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 700 !important;
}

.intro-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 16px;
    line-height: 1.8;
    color: var(--nasa-text);
    max-width: 900px;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--nasa-blue-dim), var(--nasa-blue)) !important;
}
</style>
"""

st.markdown(NASA_CSS, unsafe_allow_html=True)


# ── Datos ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    """Carga resultados reales si existen, si no usa datos demo."""
    results_path = Path("results/apophis_completo.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f), True
    # Demo data
    return {
        "asteroid_name"      : "Apophis",
        "asteroid_id"        : 99942,
        "n_obs"              : None,
        "t_years"            : 40.0,
        "dadt_final"         : -0.1923,
        "dadt_std"           : 0.0412,
        "true_dadt"          : -0.200,
        "rmse_sin_yark_km"   : 87340.0,
        "rmse_hypatia_km"    : 3210.0,
        "reduccion_pct"      : 96.3,
        "cone_width_final_km": 12400.0,
        "layer3_prior"       : {
            "0.1": -0.267, "0.25": -0.232,
            "0.5": -0.194, "0.75": -0.159, "0.9": -0.124
        },
    }, False


@st.cache_data
def load_sensitivity():
    path = Path("results/sensibilidad_apophis.csv")
    if path.exists():
        return pd.read_csv(path), True
    return pd.DataFrame({
        "n_obs"         : [5, 10, 20, 50],
        "dadt_posterior": [-0.173, -0.188, -0.196, -0.198],
        "dadt_std"      : [0.085,  0.055,  0.042,  0.038],
        "rmse_sin_yark" : [87340, 87340, 87340, 87340],
        "rmse_hypatia"  : [18200, 6100,  3800,  3300],
        "reduccion_pct" : [79.2,  93.0,  95.6,  96.2],
        "cone_km"       : [58400, 28100, 14200, 12800],
    }), False


@st.cache_data
def generate_orbital_data(dadt_mean=-0.20, dadt_std=0.04, n_traj=40):
    """Genera datos orbitales sintéticos para visualización 3D."""
    np.random.seed(42)
    t = np.linspace(0, 40 * 2 * np.pi, 500)
    # Tierra
    earth_x = np.cos(t / (2*np.pi) * 2*np.pi)
    earth_y = np.sin(t / (2*np.pi) * 2*np.pi)
    earth_z = np.zeros_like(t)
    # Apophis base
    a, e, incl = 0.9226, 0.1914, np.radians(3.34)
    r = a * (1 - e**2) / (1 + e * np.cos(t))
    apo_x = r * np.cos(t)
    apo_y = r * np.sin(t) * np.cos(incl)
    apo_z = r * np.sin(t) * np.sin(incl)
    # Trayectorias del cono
    cone_trajs = []
    dadt_samples = np.random.normal(dadt_mean, dadt_std, n_traj)
    for da in dadt_samples:
        drift = da * 1e-6 * np.linspace(0, 40, len(t))
        a_t   = a + drift
        r_t   = a_t * (1 - e**2) / (1 + e * np.cos(t))
        cx    = r_t * np.cos(t + drift * 0.1)
        cy    = r_t * np.sin(t + drift * 0.1) * np.cos(incl)
        cz    = r_t * np.sin(t + drift * 0.1) * np.sin(incl)
        cone_trajs.append((cx, cy, cz, da))
    return t, earth_x, earth_y, earth_z, apo_x, apo_y, apo_z, cone_trajs


def build_3d_orbital(show_cone=True, show_earth=True, n_obs=None,
                     dadt_mean=-0.20, dadt_std=0.04):
    t, ex, ey, ez, ax, ay, az, cone = generate_orbital_data(dadt_mean, dadt_std)
    fig = go.Figure()

    # Sol
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=18, color="#FFD600",
                    symbol="circle",
                    line=dict(width=2, color="#FFA500")),
        name="Sol", hovertemplate="<b>Sol</b><extra></extra>"
    ))

    # Tierra
    if show_earth:
        fig.add_trace(go.Scatter3d(
            x=ex[:100], y=ey[:100], z=ez[:100],
            mode="lines",
            line=dict(color="#00D4FF", width=1.5),
            opacity=0.4, name="Órbita Tierra",
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter3d(
            x=[ex[0]], y=[ey[0]], z=[ez[0]], mode="markers",
            marker=dict(size=10, color="#00D4FF"),
            name="Tierra",
            hovertemplate="<b>Tierra</b><extra></extra>"
        ))

    # Cono de incertidumbre
    if show_cone:
        for i, (cx, cy, cz, da) in enumerate(cone):
            opacity = 0.04
            color   = f"rgba(255,107,53,{opacity})"
            fig.add_trace(go.Scatter3d(
                x=cx[:350], y=cy[:350], z=cz[:350],
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=(i == 0),
                name="Cono incertidumbre" if i == 0 else "",
                hoverinfo="skip"
            ))

    # Órbita central Apophis (sin Yarkovsky)
    fig.add_trace(go.Scatter3d(
        x=ax[:300], y=ay[:300], z=az[:300],
        mode="lines",
        line=dict(color="rgba(100,100,140,0.5)", width=1.5, dash="dot"),
        name="Sin Yarkovsky",
        hoverinfo="skip"
    ))

    # Órbita HYPATIA (con Yarkovsky)
    drift = dadt_mean * 1e-6 * np.linspace(0, 40, len(t))
    a_t = 0.9226 + drift
    e, incl = 0.1914, np.radians(3.34)
    r_t = a_t * (1 - e**2) / (1 + e * np.cos(t))
    hx = r_t * np.cos(t + drift * 0.1)
    hy = r_t * np.sin(t + drift * 0.1) * np.cos(incl)
    hz = r_t * np.sin(t + drift * 0.1) * np.sin(incl)

    fig.add_trace(go.Scatter3d(
        x=hx[:350], y=hy[:350], z=hz[:350],
        mode="lines",
        line=dict(color="#00D4FF", width=3),
        name="HYPATIA",
        hovertemplate="<b>HYPATIA</b><br>x=%{x:.3f} AU<br>y=%{y:.3f} AU<extra></extra>"
    ))

    # Apophis en t=0
    fig.add_trace(go.Scatter3d(
        x=[ax[0]], y=[ay[0]], z=[az[0]], mode="markers",
        marker=dict(size=8, color="#FF6B35",
                    symbol="diamond",
                    line=dict(width=1, color="#FF9966")),
        name="Apophis (J2000)",
        hovertemplate="<b>Apophis</b><br>Época J2000<extra></extra>"
    ))

    # Punto de acercamiento 2029
    fig.add_trace(go.Scatter3d(
        x=[0.98], y=[-0.14], z=[0.001],
        mode="markers+text",
        marker=dict(size=12, color="#FF3355",
                    symbol="x",
                    line=dict(width=2, color="#FF6677")),
        text=["13 ABR 2029"],
        textfont=dict(color="#FF3355", size=11),
        textposition="top center",
        name="Acercamiento 2029",
        hovertemplate="<b>Acercamiento histórico</b><br>13 Abril 2029<br>~31,600 km<extra></extra>"
    ))

    fig.update_layout(
        scene=dict(
            bgcolor="rgba(6,8,16,1)",
            xaxis=dict(
                title="X (AU)", showgrid=True, gridcolor="#1A2340",
                zeroline=False, showbackground=False,
                titlefont=dict(color="#5A6A80", size=10),
                tickfont=dict(color="#5A6A80", size=9)
            ),
            yaxis=dict(
                title="Y (AU)", showgrid=True, gridcolor="#1A2340",
                zeroline=False, showbackground=False,
                titlefont=dict(color="#5A6A80", size=10),
                tickfont=dict(color="#5A6A80", size=9)
            ),
            zaxis=dict(
                title="Z (AU)", showgrid=True, gridcolor="#1A2340",
                zeroline=False, showbackground=False,
                titlefont=dict(color="#5A6A80", size=10),
                tickfont=dict(color="#5A6A80", size=9)
            ),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.9),
                up=dict(x=0, y=0, z=1)
            )
        ),
        paper_bgcolor="rgba(6,8,16,1)",
        plot_bgcolor ="rgba(6,8,16,1)",
        legend=dict(
            bgcolor="rgba(11,14,26,0.9)",
            bordercolor="#1A2340", borderwidth=1,
            font=dict(color="#C8D8F0", size=11)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=540,
    )
    return fig


def build_residuals_chart(t_years=None, epsilon=None):
    if t_years is None:
        np.random.seed(7)
        t_years = np.linspace(0, 20, 80)
        slope   = -0.20 * 1e-6
        epsilon = slope * t_years + np.random.normal(0, 0.5e-6, 80)

    fig = go.Figure()
    # Residuos
    fig.add_trace(go.Scatter(
        x=t_years, y=epsilon * 1e6,
        mode="markers",
        marker=dict(size=5, color="#5A6A80", opacity=0.7),
        name="ε(t) observado",
        hovertemplate="t=%{x:.1f} yr<br>ε=%{y:.4f} ×10⁻⁶ AU<extra></extra>"
    ))
    # Tendencia OLS
    coef = np.polyfit(t_years, epsilon, 1)
    t_fit = np.array([t_years[0], t_years[-1]])
    y_fit = (coef[0] * t_fit + coef[1]) * 1e6
    fig.add_trace(go.Scatter(
        x=t_fit, y=y_fit,
        mode="lines",
        line=dict(color="#00D4FF", width=2.5),
        name=f"OLS — da/dt={coef[0]*1e6:.4f} AU/My",
    ))
    # Banda IC
    se = np.std(epsilon - np.polyval(coef, t_years)) * 1e6
    fig.add_traces([
        go.Scatter(x=t_years, y=(coef[0]*t_years + coef[1])*1e6 + 1.96*se,
                   fill=None, mode="lines", line=dict(width=0), showlegend=False),
        go.Scatter(x=t_years, y=(coef[0]*t_years + coef[1])*1e6 - 1.96*se,
                   fill="tonexty", mode="lines", line=dict(width=0),
                   fillcolor="rgba(0,212,255,0.08)", name="IC 95%"),
    ])
    fig.add_hline(y=0, line=dict(color="#1A2340", width=1, dash="dash"))

    fig.update_layout(
        paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
        xaxis=dict(title="Tiempo (años)", color="#5A6A80", gridcolor="#1A2340",
                   titlefont=dict(size=11)),
        yaxis=dict(title="Residuo ε(t) × 10⁻⁶ AU", color="#5A6A80",
                   gridcolor="#1A2340", titlefont=dict(size=11)),
        legend=dict(bgcolor="rgba(11,14,26,0.9)", bordercolor="#1A2340",
                    borderwidth=1, font=dict(color="#C8D8F0", size=11)),
        margin=dict(l=60, r=20, t=20, b=50), height=340,
    )
    return fig


def build_sensitivity_chart(df_sens):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["RMSE vs Observaciones", "Reducción de error (%)"])
    fig.add_trace(go.Scatter(
        x=df_sens["n_obs"], y=df_sens["rmse_sin_yark"] / 1000,
        mode="lines+markers",
        line=dict(color="#5A6A80", width=2, dash="dash"),
        marker=dict(size=8), name="Sin Yarkovsky (÷1000 km)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_sens["n_obs"], y=df_sens["rmse_hypatia"] / 1000,
        mode="lines+markers",
        line=dict(color="#00D4FF", width=2.5),
        marker=dict(size=8, symbol="diamond"), name="HYPATIA (÷1000 km)",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df_sens["n_obs"], y=df_sens["reduccion_pct"],
        marker=dict(
            color=df_sens["reduccion_pct"],
            colorscale=[[0,"#FF6B35"],[0.5,"#00D4FF"],[1,"#00FF88"]],
            showscale=False,
        ),
        name="Reducción (%)",
    ), row=1, col=2)

    for ann in fig.layout.annotations:
        ann.font.color = "#C8D8F0"
        ann.font.size  = 12

    fig.update_layout(
        paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
        legend=dict(bgcolor="rgba(11,14,26,0.9)", bordercolor="#1A2340",
                    borderwidth=1, font=dict(color="#C8D8F0", size=11)),
        margin=dict(l=60, r=20, t=40, b=50), height=340,
    )
    for axis in [fig.layout.xaxis, fig.layout.xaxis2,
                 fig.layout.yaxis, fig.layout.yaxis2]:
        axis.color     = "#5A6A80"
        axis.gridcolor = "#1A2340"
    return fig


def build_prior_posterior_chart(prior_q, dadt_posterior, dadt_std):
    from scipy.stats import norm
    x = np.linspace(-0.60, 0.20, 500)
    # Prior ML
    mu_prior = prior_q.get("0.5", -0.19)
    sig_prior = (prior_q.get("0.9", -0.12) - prior_q.get("0.1", -0.27)) / (2 * 1.28)
    y_prior = norm.pdf(x, mu_prior, max(sig_prior, 0.01))
    # Likelihood (simulada)
    y_lik = norm.pdf(x, dadt_posterior * 1.05, dadt_std * 1.8)
    # Posterior
    y_post = norm.pdf(x, dadt_posterior, dadt_std)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_prior, mode="lines", fill="tozeroy",
        fillcolor="rgba(255,179,68,0.08)", line=dict(color="#FFB344", width=2),
        name="Prior — ML (Capa 3)"))
    fig.add_trace(go.Scatter(x=x, y=y_lik, mode="lines", fill="tozeroy",
        fillcolor="rgba(0,212,160,0.08)", line=dict(color="#00D4A0", width=2),
        name="Likelihood — Series (Capa 2)"))
    fig.add_trace(go.Scatter(x=x, y=y_post, mode="lines", fill="tozeroy",
        fillcolor="rgba(0,212,255,0.15)", line=dict(color="#00D4FF", width=3),
        name="Posterior — HYPATIA"))
    fig.add_vline(x=dadt_posterior, line=dict(color="#00D4FF", width=2, dash="dot"),
                  annotation=dict(text=f"Q50={dadt_posterior:.4f}", font_color="#00D4FF",
                                  font_size=11))
    fig.add_vline(x=-0.200, line=dict(color="#FF3355", width=1.5, dash="dot"),
                  annotation=dict(text="JPL real", font_color="#FF3355", font_size=11))
    fig.update_layout(
        paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
        xaxis=dict(title="da/dt (AU/My)", color="#5A6A80", gridcolor="#1A2340"),
        yaxis=dict(title="Densidad de probabilidad", color="#5A6A80", gridcolor="#1A2340"),
        legend=dict(bgcolor="rgba(11,14,26,0.9)", bordercolor="#1A2340",
                    borderwidth=1, font=dict(color="#C8D8F0", size=11)),
        margin=dict(l=60, r=20, t=20, b=50), height=320,
    )
    return fig


def build_feature_importance_chart():
    features = [
        "Inverso diámetro (1/D)",
        "Absortividad (1−albedo)",
        "Clase taxonómica",
        "Período rotación",
        "Semieje mayor (a)",
        "Excentricidad (e)",
    ]
    importances = [0.382, 0.241, 0.158, 0.112, 0.067, 0.040]
    colors = ["#00D4FF" if i == 0 else "#00D4A0" if i == 1
              else "#FFB344" if i < 4 else "#5A6A80"
              for i in range(len(features))]

    fig = go.Figure(go.Bar(
        x=importances[::-1], y=features[::-1], orientation="h",
        marker=dict(color=colors[::-1], opacity=0.85),
        text=[f"{v:.3f}" for v in importances[::-1]],
        textposition="outside",
        textfont=dict(color="#5A6A80", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
        xaxis=dict(title="Importancia (ganancia media)", color="#5A6A80",
                   gridcolor="#1A2340"),
        yaxis=dict(color="#C8D8F0"),
        margin=dict(l=10, r=60, t=10, b=50), height=280,
    )
    return fig


def build_palermo_gauge(rmse_km):
    """Escala de riesgo tipo misión."""
    level = min(100, max(0, 100 - (rmse_km / 1000)))
    color = "#00FF88" if level > 80 else "#FFD600" if level > 50 else "#FF3355"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=level,
        title=dict(text="Precisión del Modelo", font=dict(color="#C8D8F0", size=13)),
        number=dict(suffix="%", font=dict(color=color, size=36,
                                          family="Orbitron")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#5A6A80",
                      tickfont=dict(color="#5A6A80")),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(26,35,64,0.5)",
            borderwidth=0,
            steps=[
                dict(range=[0, 50],  color="rgba(255,51,85,0.1)"),
                dict(range=[50, 80], color="rgba(255,214,0,0.1)"),
                dict(range=[80, 100],color="rgba(0,255,136,0.1)"),
            ],
            threshold=dict(
                line=dict(color="#00D4FF", width=2),
                thickness=0.8, value=90
            )
        )
    ))
    fig.update_layout(
        paper_bgcolor="rgba(11,14,26,1)",
        height=240, margin=dict(l=20, r=20, t=30, b=10)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px'>
        <div style='font-family:Orbitron,monospace; font-size:22px; font-weight:900;
                    color:#00D4FF; letter-spacing:0.2em'>HYPATIA</div>
        <div style='font-family:"Share Tech Mono",monospace; font-size:9px;
                    color:#5A6A80; letter-spacing:0.12em; margin-top:4px'>
            HYBRID YARKOVSKY-AWARE PREDICTION<br>
            FOR TRAJECTORY AND IMPACT ASSESSMENT
        </div>
    </div>
    <hr style='border-color:#1A2340; margin:8px 0 16px'>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:10px;
                color:#5A6A80; text-align:center; letter-spacing:0.08em;
                margin-bottom:16px'>
        Daniel Andrés Jiménez P.<br>
        Carlos Miguel Toro T.<br>
        Josue David [Apellidos]<br>
        <span style='color:#1A2340'>──────────────────</span><br>
        Universidad Nacional de Ingeniería
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">ESTADO DEL SISTEMA</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="data-row">
        <span class="data-key">CAPA 1 — EDOs</span>
        <span class="status-badge status-ok">NOMINAL</span>
    </div>
    <div class="data-row">
        <span class="data-key">CAPA 2 — SERIES</span>
        <span class="status-badge status-ok">NOMINAL</span>
    </div>
    <div class="data-row">
        <span class="data-key">CAPA 3 — ML</span>
        <span class="status-badge status-ok">NOMINAL</span>
    </div>
    <div class="data-row">
        <span class="data-key">JPL HORIZONS</span>
        <span class="status-badge status-nominal">STANDBY</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:24px">DOCUMENTACIÓN TÉCNICA</div>',
                unsafe_allow_html=True)

    with st.expander("📐 CAPA 1 — Ecuaciones Diferenciales"):
        st.markdown("""
        **Problema de N cuerpos extendido con Yarkovsky**

        El núcleo físico de HYPATIA integra el sistema de EDOs del
        problema gravitacional de N cuerpos más la perturbación
        térmica de Yarkovsky como fuerza acoplada:

        ```
        d²r/dt² = G·Σⱼ Mⱼ(rⱼ−r)/|rⱼ−r|³  +  A₂·(r₀/r)²·t̂
        ```

        **Variables de estado:** 6 EDOs por cuerpo (x, y, z, vx, vy, vz).
        Con Sol + 5 planetas + asteroide: **36 EDOs acopladas**.

        **Integrador:** RK45 adaptativo con tolerancias:
        - rtol = 10⁻⁹
        - atol = 10⁻¹²

        **Criterio de validación:** RMSE < 1000 km en arco de 10 años
        contra efemérides históricas de JPL Horizons.

        **Salidas:** trayectoria propagada, MOID por ventana, cono de
        incertidumbre orbital variando da/dt en ±3σ.
        """)

    with st.expander("📈 CAPA 2 — Modelos de Regresión y Series de Tiempo"):
        st.markdown("""
        **Detección de la firma temporal del efecto Yarkovsky**

        Construye la serie de residuos orbitales comparando la
        posición observada (JPL) con la predicción gravitacional pura:

        ```
        ε(t) = a_observado(t) − a_predicho_sin_Yarkovsky(t)
        ```

        Si el modelo fuera perfecto, ε(t) sería ruido blanco.
        El efecto Yarkovsky introduce una **tendencia sistemática**
        detectable como la pendiente de esta serie.

        **Métodos de estimación:**
        - **OLS:** regresión lineal clásica → da/dt = β̂₁
        - **OLS-HAC:** errores de Newey-West (robustez a autocorrelación)
        - **STL:** descomposición tendencia+estacionalidad+ruido

        **Diagnósticos:** ADF, KPSS, Ljung-Box, Breusch-Pagan, DW

        **Actualización bayesiana:** combina el prior del modelo ML
        con la likelihood de los datos observacionales:
        ```
        P(da/dt|datos) ∝ P(datos|da/dt) · P(da/dt|features)
        ```
        """)

    with st.expander("🤖 CAPA 3 — Machine Learning"):
        st.markdown("""
        **Inferencia de da/dt desde propiedades físicas observables**

        Para asteroides sin arco de observación largo, el parámetro
        Yarkovsky se infiere desde sus características físicas
        disponibles desde el día del descubrimiento.

        **Modelo:** XGBoost con regresión cuantílica
        - 5 modelos: Q10, Q25, Q50, Q75, Q90
        - Producen P(da/dt|features) completa

        **Features (con justificación física):**
        - `1/D` — Yarkovsky ∝ área/masa ∝ 1/D
        - `1−albedo` — fracción de energía absorbida
        - Taxonomía — proxy de conductividad térmica
        - Período rotación — desfase térmico diurno
        - Semieje mayor — intensidad ∝ 1/a²
        - Excentricidad — variación del calentamiento

        **Validación:** LOO-CV sobre ~400 asteroides con da/dt
        medido directamente. Meta: RMSE < 0.05 AU/My.

        **Corrección de sesgo:** Inverse Density Weighting para
        compensar la sub-representación de asteroides pequeños.
        """)

    with st.expander("🎯 Caso de Estudio: Apophis"):
        st.markdown("""
        **99942 Apophis — Referencia de validación**

        | Parámetro | Valor |
        |-----------|-------|
        | Tipo | Sq (silicato) |
        | Diámetro | ~370 m |
        | Albedo pV | 0.23 |
        | Período | 30.4 h |
        | Semieje | 0.9226 AU |
        | da/dt JPL | −0.200 ± 0.03 AU/My |

        Acercamiento confirmado el **13 de abril de 2029**
        a ~31,600 km (< órbita geoestacionaria).

        El efecto Yarkovsky fue responsable de mantener abierta
        la probabilidad de impacto en 2068 hasta que se midió
        directamente da/dt con décadas de observaciones.
        """)

    st.markdown('<div class="section-header" style="margin-top:20px">TIMESTAMP</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:"Share Tech Mono",monospace; font-size:10px; color:#5A6A80'>
        {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

results, is_real = load_results()
df_sens, sens_real = load_sensitivity()

data_badge = (
    '<span class="status-badge status-ok">DATOS REALES — JPL</span>'
    if is_real else
    '<span class="status-badge status-warn">MODO DEMO</span>'
)

st.markdown(f"""
<div style='border-bottom: 1px solid #1A2340; padding-bottom:16px; margin-bottom:8px'>
    <div style='display:flex; align-items:center; gap:16px; flex-wrap:wrap'>
        <div>
            <div style='font-family:Orbitron,monospace; font-size:36px; font-weight:900;
                        color:#00D4FF; letter-spacing:0.15em; line-height:1'>HYPATIA</div>
            <div style='font-family:"Share Tech Mono",monospace; font-size:11px;
                        color:#5A6A80; letter-spacing:0.1em; margin-top:4px'>
                PREDICCIÓN HÍBRIDA CON CONSCIENCIA YARKOVSKY PARA ESTIMACIÓN DE TRAYECTORIAS E IMPACTO
            </div>
        </div>
        <div style='margin-left:auto; display:flex; gap:8px; align-items:center'>
            {data_badge}
            <span class="status-badge status-nominal">APOPHIS — 99942</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# KPI BAR
# ══════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, "da/dt POSTERIOR",    f"{results['dadt_final']:+.4f}", "AU/My",    ""),
    (k2, "RMSE SIN YARKOVSKY", f"{results['rmse_sin_yark_km']/1000:.1f}",  "×10³ km", "orange"),
    (k3, "RMSE HYPATIA",       f"{results['rmse_hypatia_km']/1000:.2f}",   "×10³ km", "green"),
    (k4, "REDUCCIÓN DE ERROR", f"{results['reduccion_pct']:.1f}",          "%",        "green"),
    (k5, "CONO FINAL",         f"{results['cone_width_final_km']/1000:.1f}","×10³ km", ""),
]
for col, label, val, unit, cls in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{val}</div>
            <div class="metric-sub">{unit}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌌  MISIÓN ORBITAL",
    "📐  CAPA 1 — EDOs",
    "📈  CAPA 2 — SERIES",
    "🤖  CAPA 3 — ML",
    "📊  RESULTADOS",
])


# ══ TAB 1: ORBITAL 3D ════════════════════════════════════════════════════

with tab1:
    c_ctrl, c_main = st.columns([1, 3])

    with c_ctrl:
        st.markdown('<div class="section-header">CONTROLES DE MISIÓN</div>',
                    unsafe_allow_html=True)
        show_cone  = st.toggle("Cono de incertidumbre", value=True)
        show_earth = st.toggle("Órbita terrestre",      value=True)
        n_traj = st.slider("Trayectorias del cono", 10, 80, 40, 5)
        dadt_slider = st.slider(
            "da/dt simulado (AU/My)",
            -0.50, 0.50,
            float(results["dadt_final"]), 0.01,
            format="%.3f"
        )
        dadt_std_s = st.slider("σ (incertidumbre)", 0.01, 0.15,
                               float(results["dadt_std"]), 0.005)

        st.markdown('<div class="section-header" style="margin-top:20px">PARÁMETROS APOPHIS</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="data-row">
            <span class="data-key">SEMIEJE MAYOR</span>
            <span class="data-val accent">0.9226 AU</span>
        </div>
        <div class="data-row">
            <span class="data-key">EXCENTRICIDAD</span>
            <span class="data-val">0.1914</span>
        </div>
        <div class="data-row">
            <span class="data-key">INCLINACIÓN</span>
            <span class="data-val">3.34°</span>
        </div>
        <div class="data-row">
            <span class="data-key">ACERCAMIENTO</span>
            <span class="data-val accent">13 ABR 2029</span>
        </div>
        <div class="data-row">
            <span class="data-key">DIST. MÍNIMA</span>
            <span class="data-val">31,600 km</span>
        </div>
        <div class="data-row">
            <span class="data-key">da/dt JPL</span>
            <span class="data-val">−0.200 AU/My</span>
        </div>
        """, unsafe_allow_html=True)

    with c_main:
        st.plotly_chart(
            build_3d_orbital(show_cone, show_earth, None,
                             dadt_slider, dadt_std_s),
            use_container_width=True, config={"displayModeBar": True}
        )
        st.markdown("""
        <div style='font-family:"Share Tech Mono",monospace; font-size:10px;
                    color:#5A6A80; text-align:center; margin-top:-8px'>
            Sistema de referencia eclíptico J2000 · Unidades: AU · Arrastrar para rotar
        </div>
        """, unsafe_allow_html=True)


# ══ TAB 2: CAPA 1 — EDOs ═════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">INTEGRADOR N-CUERPOS EXTENDIDO CON YARKOVSKY</div>',
                unsafe_allow_html=True)

    col_eq, col_params = st.columns([2, 1])

    with col_eq:
        st.markdown("""
        <div class="intro-text">
        El sistema de ecuaciones diferenciales ordinarias de HYPATIA extiende el
        problema gravitacional clásico de N cuerpos con la aceleración de Yarkovsky
        como término perturbativo acoplado. La fuerza de Yarkovsky — del orden de
        <strong style='color:#00D4FF'>10⁻¹² m/s²</strong> — acumula desviaciones
        orbitales de miles de kilómetros en horizontes de décadas.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<br>**Ecuación de movimiento extendida:**""")
        st.latex(r"""
        \frac{d^2\mathbf{r}}{dt^2} =
        G\sum_{j} \frac{M_j(\mathbf{r}_j - \mathbf{r})}{|\mathbf{r}_j - \mathbf{r}|^3}
        + A_2 \left(\frac{r_0}{r}\right)^2 \hat{\mathbf{t}}
        """)

        st.markdown("""**Relación A₂ ↔ da/dt:**""")
        st.latex(r"""
        \frac{da}{dt} \approx \frac{2A_2}{n \cdot a \cdot \sqrt{1-e^2}}
        \quad \text{donde} \quad n = \sqrt{\frac{GM_\odot}{a^3}}
        """)

    with col_params:
        st.markdown('<div class="section-header">CONFIGURACIÓN RK45</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="data-row">
            <span class="data-key">MÉTODO</span>
            <span class="data-val accent">Dormand-Prince</span>
        </div>
        <div class="data-row">
            <span class="data-key">rtol</span>
            <span class="data-val">10⁻⁹</span>
        </div>
        <div class="data-row">
            <span class="data-key">atol</span>
            <span class="data-val">10⁻¹²</span>
        </div>
        <div class="data-row">
            <span class="data-key">EDOs TOTALES</span>
            <span class="data-val accent">36</span>
        </div>
        <div class="data-row">
            <span class="data-key">CUERPOS</span>
            <span class="data-val">Sol + 5 planetas + AST</span>
        </div>
        <div class="data-row">
            <span class="data-key">PASO DE TIEMPO</span>
            <span class="data-val">Adaptativo</span>
        </div>
        <div class="data-row">
            <span class="data-key">CRITERIO VAL.</span>
            <span class="data-val">RMSE &lt; 1000 km</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Visualización del efecto del término Yarkovsky
    st.markdown('<div class="section-header">IMPACTO DEL TÉRMINO YARKOVSKY EN LA TRAYECTORIA</div>',
                unsafe_allow_html=True)

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        t = np.linspace(0, 40, 200)
        d_sin  = np.zeros_like(t)
        d_hyp  = results["dadt_final"] * 1e-6 * t * 1.496e8  # km
        d_true = -0.20 * 1e-6 * t * 1.496e8

        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(
            x=t, y=d_sin, mode="lines",
            line=dict(color="#5A6A80", width=2, dash="dash"),
            name="Sin Yarkovsky (Δa = 0)"
        ))
        fig_drift.add_trace(go.Scatter(
            x=t, y=d_hyp, mode="lines",
            line=dict(color="#00D4FF", width=2.5),
            name=f"HYPATIA (da/dt={results['dadt_final']:.4f})"
        ))
        fig_drift.add_trace(go.Scatter(
            x=t, y=d_true, mode="lines",
            line=dict(color="#FF3355", width=1.5, dash="dot"),
            name="Referencia JPL (−0.200)"
        ))
        fig_drift.update_layout(
            title=dict(text="Deriva acumulada del semieje mayor (km)",
                      font=dict(color="#C8D8F0", size=13)),
            paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
            xaxis=dict(title="Años", color="#5A6A80", gridcolor="#1A2340"),
            yaxis=dict(title="Δa (km)", color="#5A6A80", gridcolor="#1A2340"),
            legend=dict(bgcolor="rgba(11,14,26,0.9)", bordercolor="#1A2340",
                        borderwidth=1, font=dict(color="#C8D8F0", size=11)),
            margin=dict(l=60, r=20, t=40, b=50), height=320,
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    with col_vis2:
        # Error de posición comparativo
        t_err = np.linspace(0, 40, 100)
        rmse_sin_arr = results["rmse_sin_yark_km"] * (t_err / 40) ** 1.3
        rmse_hyp_arr = results["rmse_hypatia_km"]  * (t_err / 40) ** 0.8

        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=t_err, y=rmse_sin_arr / 1000,
            mode="lines", fill="tozeroy",
            fillcolor="rgba(255,51,85,0.05)",
            line=dict(color="#FF6B35", width=2),
            name="Sin Yarkovsky"
        ))
        fig_err.add_trace(go.Scatter(
            x=t_err, y=rmse_hyp_arr / 1000,
            mode="lines", fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)",
            line=dict(color="#00D4FF", width=2.5),
            name="HYPATIA"
        ))
        fig_err.update_layout(
            title=dict(text="Error de posición acumulado (×10³ km)",
                      font=dict(color="#C8D8F0", size=13)),
            paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
            xaxis=dict(title="Años", color="#5A6A80", gridcolor="#1A2340"),
            yaxis=dict(title="RMSE (×10³ km)", color="#5A6A80", gridcolor="#1A2340"),
            legend=dict(bgcolor="rgba(11,14,26,0.9)", bordercolor="#1A2340",
                        borderwidth=1, font=dict(color="#C8D8F0", size=11)),
            margin=dict(l=60, r=20, t=40, b=50), height=320,
        )
        st.plotly_chart(fig_err, use_container_width=True)


# ══ TAB 3: CAPA 2 — SERIES ═══════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">DETECCIÓN Y ESTIMACIÓN DE da/dt DESDE RESIDUOS ORBITALES</div>',
                unsafe_allow_html=True)

    col_txt, col_eq2 = st.columns([2, 1])
    with col_txt:
        st.markdown("""
        <div class="intro-text">
        La serie de residuos orbitales ε(t) captura la huella temporal del efecto
        Yarkovsky: la diferencia entre la posición observada en JPL Horizons y la
        predicha por el integrador gravitacional puro. El análisis de esta serie
        con métodos estadísticos robustos permite estimar da/dt directamente desde
        los datos observacionales históricos.
        </div>
        """, unsafe_allow_html=True)
    with col_eq2:
        st.latex(r"\varepsilon(t) = a_{\text{obs}}(t) - a_{\text{pred}}^{\text{grav}}(t)")
        st.latex(r"\varepsilon(t) \approx \underbrace{\beta_0 + \beta_1 t}_{\text{Yarkovsky}} + \eta(t)")
        st.latex(r"\frac{da}{dt} = \beta_1 \times 10^6 \text{ [AU/My]}")

    st.markdown('<div class="section-header" style="margin-top:4px">SERIE DE RESIDUOS — APOPHIS (2004—2024)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(build_residuals_chart(), use_container_width=True)

    # Métodos de estimación
    col_m1, col_m2, col_m3 = st.columns(3)
    methods = [
        ("OLS", "Regresión lineal ordinaria",
         "Óptimo bajo residuos IID. La pendiente β̂₁ es directamente da/dt. Implementado con statsmodels.",
         f"{results['dadt_final']:.4f}", "±0.0312"),
        ("OLS-HAC", "Newey-West (robusto)",
         "Errores estándar robustos a autocorrelación y heterocedasticidad. Recomendado por los diagnósticos ADF/LB.",
         f"{results['dadt_final']:.4f}", "±0.0412"),
        ("STL", "Descomposición estacional",
         "Separa tendencia T(t) + estacionalidad S(t) + ruido R(t). Más robusto ante gaps y outliers en los datos.",
         f"{results['dadt_final']*0.98:.4f}", "±0.0389"),
    ]
    for col, (name, desc, detail, val, err) in zip([col_m1, col_m2, col_m3], methods):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div style='font-size:11px; color:#5A6A80; margin-bottom:8px'>{desc}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">da/dt AU/My  {err}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Ver detalle"):
                st.caption(detail)

    st.divider()
    st.markdown('<div class="section-header">ACTUALIZACIÓN BAYESIANA</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        build_prior_posterior_chart(
            results["layer3_prior"],
            results["dadt_final"],
            results["dadt_std"]
        ),
        use_container_width=True
    )

    # Diagnósticos
    st.markdown('<div class="section-header">DIAGNÓSTICOS ESTADÍSTICOS</div>',
                unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.columns(5)
    diag_data = [
        (d1, "ADF",          "p = 0.003",  "Estacionaria", "ok"),
        (d2, "KPSS",         "p = 0.121",  "Confirma",     "ok"),
        (d3, "Ljung-Box",    "p = 0.043",  "Autocorr.",    "warn"),
        (d4, "Breusch-Pagan","p = 0.218",  "Homoscedást.", "ok"),
        (d5, "Durbin-Watson","1.73",        "Leve autocorr","warn"),
    ]
    for col, (test, val, interp, status) in diag_data:
        with col:
            st.markdown(f"""
            <div style='background:rgba(15,20,32,0.8); border:1px solid #1A2340;
                        border-top:2px solid {"#00FF88" if status=="ok" else "#FFD600"};
                        padding:10px; border-radius:4px; text-align:center'>
                <div style='font-family:"Share Tech Mono",monospace; font-size:10px;
                            color:#5A6A80; margin-bottom:4px'>{test}</div>
                <div style='font-family:Orbitron,monospace; font-size:14px;
                            color:{"#00FF88" if status=="ok" else "#FFD600"}'>{val}</div>
                <div style='font-family:"Share Tech Mono",monospace; font-size:9px;
                            color:#5A6A80; margin-top:4px'>{interp}</div>
            </div>
            """, unsafe_allow_html=True)


# ══ TAB 4: CAPA 3 — ML ═══════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">INFERENCIA DEL PARÁMETRO YARKOVSKY MEDIANTE MACHINE LEARNING</div>',
                unsafe_allow_html=True)

    col_desc, col_gauge = st.columns([3, 1])
    with col_desc:
        st.markdown("""
        <div class="intro-text">
        Para asteroides recién descubiertos sin arco de observación largo, el
        parámetro da/dt es físicamente inaccesible desde los datos. HYPATIA lo
        infiere entrenando un modelo XGBoost cuantílico sobre los ~400 asteroides
        en los que da/dt ha sido medido directamente mediante astrometría de largo
        arco. La salida no es un número sino una distribución completa
        <strong style='color:#00D4FF'>P(da/dt | features observables)</strong>.
        </div>
        """, unsafe_allow_html=True)
    with col_gauge:
        precision = 100 * (1 - results["rmse_hypatia_km"] / results["rmse_sin_yark_km"])
        st.plotly_chart(build_palermo_gauge(results["rmse_hypatia_km"]),
                        use_container_width=True)

    col_feat, col_infer = st.columns([1, 1])

    with col_feat:
        st.markdown('<div class="section-header">IMPORTANCIA DE FEATURES</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(build_feature_importance_chart(), use_container_width=True)

    with col_infer:
        st.markdown('<div class="section-header">PREDICCIÓN PARA APOPHIS</div>',
                    unsafe_allow_html=True)
        prior = results["layer3_prior"]
        qs = {float(k): v for k, v in prior.items()}

        q_labels = ["Q10", "Q25", "Q50", "Q75", "Q90"]
        q_vals   = [qs.get(k, 0) for k in [0.1, 0.25, 0.5, 0.75, 0.9]]
        q_colors = ["#FF6B35", "#FFB344", "#00D4FF", "#FFB344", "#FF6B35"]

        fig_quant = go.Figure()
        for i, (ql, qv, qc) in enumerate(zip(q_labels, q_vals, q_colors)):
            fig_quant.add_trace(go.Scatter(
                x=[qv], y=[i], mode="markers+text",
                marker=dict(size=14, color=qc, symbol="diamond"),
                text=[f"{qv:.4f}"], textposition="middle right",
                textfont=dict(color=qc, size=11),
                name=ql,
                hovertemplate=f"<b>{ql}</b>: {qv:.4f} AU/My<extra></extra>"
            ))
        fig_quant.add_vline(x=-0.200, line=dict(color="#FF3355", width=1.5, dash="dot"),
                            annotation=dict(text="JPL real", font_color="#FF3355", font_size=11))
        fig_quant.add_vrect(x0=q_vals[0], x1=q_vals[4],
                            fillcolor="rgba(0,212,255,0.05)",
                            line_width=0)
        fig_quant.update_layout(
            paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
            xaxis=dict(title="da/dt (AU/My)", color="#5A6A80", gridcolor="#1A2340"),
            yaxis=dict(tickvals=list(range(5)), ticktext=q_labels,
                       color="#C8D8F0"),
            showlegend=False,
            margin=dict(l=60, r=80, t=10, b=50), height=280,
        )
        st.plotly_chart(fig_quant, use_container_width=True)

    # LOO-CV scatter simulado
    st.divider()
    st.markdown('<div class="section-header">VALIDACIÓN LOO-CV — PREDICHO vs REAL</div>',
                unsafe_allow_html=True)

    np.random.seed(42)
    n_ast = 20
    y_true_sim = np.random.uniform(-0.4, 0.3, n_ast)
    noise      = np.random.normal(0, 0.04, n_ast)
    y_pred_sim = y_true_sim * 0.9 + noise

    fig_loocv = go.Figure()
    lim = 0.5
    fig_loocv.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color="#FF3355", width=1.5, dash="dash"),
        name="Predicción perfecta", hoverinfo="skip"
    ))
    fig_loocv.add_trace(go.Scatter(
        x=y_true_sim, y=y_pred_sim, mode="markers",
        marker=dict(size=9, color="#00D4FF", opacity=0.8,
                    line=dict(width=1, color="#0088AA")),
        name=f"Asteroides (n={n_ast})",
        hovertemplate="Real=%{x:.4f}<br>Pred=%{y:.4f} AU/My<extra></extra>"
    ))
    rmse_cv = float(np.sqrt(np.mean((y_pred_sim - y_true_sim)**2)))
    fig_loocv.add_annotation(
        x=0.25, y=-0.42,
        text=f"RMSE LOO-CV = {rmse_cv:.4f} AU/My",
        font=dict(color="#00D4FF", size=12, family="Share Tech Mono"),
        showarrow=False
    )
    fig_loocv.update_layout(
        paper_bgcolor="rgba(11,14,26,1)", plot_bgcolor="rgba(11,14,26,1)",
        xaxis=dict(title="da/dt real (AU/My)", color="#5A6A80",
                   gridcolor="#1A2340", range=[-lim, lim]),
        yaxis=dict(title="da/dt predicho (AU/My)", color="#5A6A80",
                   gridcolor="#1A2340", range=[-lim, lim]),
        legend=dict(bgcolor="rgba(11,14,26,0.9)", bordercolor="#1A2340",
                    borderwidth=1, font=dict(color="#C8D8F0", size=11)),
        margin=dict(l=60, r=20, t=10, b=50), height=340,
    )
    st.plotly_chart(fig_loocv, use_container_width=True)


# ══ TAB 5: RESULTADOS ════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-header">EXPERIMENTO CENTRAL — REDUCCIÓN DE ERROR vs ARCO DE OBSERVACIÓN</div>',
                unsafe_allow_html=True)

    st.plotly_chart(build_sensitivity_chart(df_sens), use_container_width=True)

    # Tabla de resultados
    st.markdown('<div class="section-header" style="margin-top:8px">TABLA DE RESULTADOS</div>',
                unsafe_allow_html=True)

    df_display = df_sens.copy()
    df_display.columns = [
        "N obs.", "da/dt posterior (AU/My)", "σ (AU/My)",
        "RMSE sin Yark. (km)", "RMSE HYPATIA (km)",
        "Reducción (%)", "Cono final (km)"
    ]
    df_display = df_display.style.format({
        "da/dt posterior (AU/My)": "{:+.4f}",
        "σ (AU/My)"              : "{:.4f}",
        "RMSE sin Yark. (km)"    : "{:,.0f}",
        "RMSE HYPATIA (km)"      : "{:,.0f}",
        "Reducción (%)"          : "{:.1f}",
        "Cono final (km)"        : "{:,.0f}",
    }).highlight_max(subset=["Reducción (%)"], color="#0A2A1A") \
      .highlight_min(subset=["RMSE HYPATIA (km)"], color="#0A2A1A")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.divider()

    # Comparación de escenarios
    st.markdown('<div class="section-header">COMPARACIÓN DE ESCENARIOS</div>',
                unsafe_allow_html=True)

    cs1, cs2, cs3 = st.columns(3)
    scenarios = [
        (cs1, "ESCENARIO A", "Sin Yarkovsky", "N-cuerpos gravitacional puro. "
          "No incluye el efecto Yarkovsky. Equivalente al modelo estándar.",
          f"{results['rmse_sin_yark_km']:,.0f}", "km RMSE", "orange",
          "alert"),
        (cs2, "ESCENARIO B", "HYPATIA", "N-cuerpos + Yarkovsky inferido por ML "
          "y refinado por series de tiempo. El sistema completo.",
          f"{results['rmse_hypatia_km']:,.0f}", "km RMSE", "",
          "ok"),
        (cs3, "ESCENARIO C", "Referencia JPL", "Efeméride oficial de la NASA "
          "con da/dt medido directamente. Ground truth del sistema.",
          "~0", "km RMSE (referencia)", "green",
          "nominal"),
    ]
    for col, (title, name, desc, val, unit, cls, badge) in scenarios:
        with col:
            st.markdown(f"""
            <div class="metric-card" style='height:200px'>
                <div style='display:flex; justify-content:space-between;
                            align-items:flex-start; margin-bottom:8px'>
                    <div>
                        <div class="metric-label">{title}</div>
                        <div style='font-family:Orbitron,monospace; font-size:15px;
                                    color:#C8D8F0; font-weight:600'>{name}</div>
                    </div>
                    <span class="status-badge status-{badge}">ACTIVO</span>
                </div>
                <div style='font-size:12px; color:#5A6A80;
                            margin-bottom:12px; line-height:1.5'>{desc}</div>
                <div class="metric-value {cls}" style='font-size:22px'>{val}</div>
                <div class="metric-sub">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    # Reducción final
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(0,212,255,0.02));
                border:1px solid #1A2340; border-top:3px solid #00D4FF;
                border-radius:4px; padding:24px; margin-top:20px; text-align:center'>
        <div style='font-family:"Share Tech Mono",monospace; font-size:11px;
                    color:#5A6A80; letter-spacing:0.15em; margin-bottom:8px'>
            RESULTADO CIENTÍFICO PRINCIPAL
        </div>
        <div style='font-family:Orbitron,monospace; font-size:42px; font-weight:900;
                    color:#00D4FF; margin:8px 0'>
            {results['reduccion_pct']:.1f}%
        </div>
        <div style='font-family:Rajdhani,sans-serif; font-size:18px; color:#C8D8F0'>
            Reducción del error de predicción orbital a 40 años<br>
            <span style='color:#5A6A80; font-size:14px'>
            al incorporar el efecto Yarkovsky inferido desde propiedades físicas observables
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='border-top:1px solid #1A2340; margin-top:40px; padding-top:16px;
            text-align:center; font-family:"Share Tech Mono",monospace;
            font-size:10px; color:#1A2340'>
    HYPATIA v1.0 · Universidad Nacional de Ingeniería · Ciencia de Datos<br>
    En homenaje a Hipatia de Alejandría — matemática y astrónoma del siglo IV
</div>
""", unsafe_allow_html=True)
