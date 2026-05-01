# Hypatia_Yarkovsky_Aware_Prediction_for_Trajectory_and_impact_Assessment
HYPATIA: Sistema Híbrido de Predicción de Trayectorias de Objetos Cercanos a la Tierra



![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-En%20desarrollo-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

Sistema computacional híbrido que predice trayectorias de asteroides cercanos a la Tierra incorporando el **efecto Yarkovsky** — una perturbación térmica ignorada en modelos gravitacionales clásicos que acumula desvíos orbitales de miles de kilómetros en horizontes de décadas.

---

## ¿Por qué HYPATIA?

Los modelos orbitales estándar omiten el efecto Yarkovsky porque su parámetro (`da/dt`) es desconocido para la mayoría de asteroides recién descubiertos. HYPATIA lo infiere desde propiedades físicas observables y lo incorpora directamente al integrador orbital, reduciendo el cono de incertidumbre en predicciones de largo plazo.

El sistema se valida sobre el asteroide **Apophis**, cuyo acercamiento confirmado en abril de 2029 a menos de 32.000 km de la Tierra lo convierte en el caso de referencia ideal.

---

## Arquitectura

HYPATIA opera en tres capas integradas:

| Capa | Componente | Descripción |
|------|-----------|-------------|
| 1 | **EDOs** | Integrador N-cuerpos extendido con Yarkovsky como fuerza perturbativa (RK45) |
| 2 | **Series de tiempo** | Estimación de `da/dt` desde residuos orbitales históricos (OLS, STL, Bayesiano) |
| 3 | **Machine Learning** | Inferencia de `da/dt` desde propiedades físicas observables (XGBoost cuantílico) |

La salida del modelo ML alimenta como prior bayesiano a la capa de series de tiempo, que a su vez parametriza el integrador orbital.

---

## Instalación

```bash
git clone https://github.com/usuario/hypatia.git
cd hypatia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # agregar NASA API Key
```

---

## Uso

```bash
# Pipeline completo sobre Apophis
python src/pipeline.py --target 99942 --n-obs 10 --years 40

# Dashboard interactivo
streamlit run dashboard/app.py
```

---

## Equipo

Desarrollado en la **Universidad Nacional de Ingeniería (UNI)** como proyecto integrado de Ciencia de Datos.

| | Integrante |
| | Josue David (apellidos)
| | Daniel Andres Jimenez Povea 
| | Carlos Manuel Toro Torres 

---

## Referencias principales

- Farnocchia et al. (2013). *Near Earth Asteroids with measurable Yarkovsky effect.* Icarus, 224(1).
- Vokrouhlický et al. (2015). *The Yarkovsky and YORP effects.* Asteroids IV, UAPress.
- Dormand & Prince (1980). *A family of embedded Runge-Kutta formulae.* JCAM, 6(1).

---

<div align="center">
<sub>En homenaje a Hipatia de Alejandría — matemática y astrónoma del siglo IV</sub>
</div>