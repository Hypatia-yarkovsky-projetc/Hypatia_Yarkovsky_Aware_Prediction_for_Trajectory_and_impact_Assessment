"""
pipeline.py
Pipeline maestro de HYPATIA: integra las tres capas en un flujo único
que produce la predicción de trayectoria y riesgo de impacto.
Flujo completo:
1. Capa 3 (ML)          → P(da/dt | features físicas)   [prior bayesiano]
2. Capa 2 (Series)      → P(da/dt | datos históricos)   [posterior]
3. Capa 1 (EDOs)        → Trayectoria con Yarkovsky     [cono orbital]
4. Comparación          → Escenarios A/B/C vs JPL       [resultado científico]
Experimento central HYPATIA:
Para N_obs ∈ {5, 10, 20, completo}:
A: N-cuerpos sin Yarkovsky          → RMSE_A
B: HYPATIA con Yarkovsky inferido   → RMSE_B
C: Referencia JPL (ground truth)
Resultado: reducción (RMSE_A - RMSE_B) / RMSE_A × 100 %
Uso:
python src/pipeline.py --target 99942 --n-obs 10 --years 40
"""
import numpy as np
import pandas as pd
import argparse
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Imports de las tres capas
from src.layer3_ml.pipeline import run_layer3, run_layer3_offline
from src.layer2_ts.pipeline import run_layer2, run_layer2_offline
from src.layer1_ode import (
    get_initial_conditions, pack_state_vector,
    propagate_from_state,
    jd_to_iso, iso_to_jd,
)
from src.layer1_ode.moid import generate_uncertainty_cone, cone_width_at_year
from src.layer1_ode.validation import fetch_ephemeris_arc, compare_scenarios, compute_position_errors
from src.layer1_ode.constants import DEFAULT_PERTURBERS
from src.layer1_ode.yarkovsky import dadt_to_A2

KM_PER_AU = 1.495978707e8

# Configuración de objetos de estudio
ASTEROID_CONFIGS = {
    99942: {
        "name": "Apophis",
        "a_AU": 0.9226,
        "ecc": 0.1914,
        "epoch_start": "2004-06-19",
        "epoch_valid": "2014-01-01",
        "epoch_end": "2024-01-01",
        "true_dadt": -0.200,
        "features": {
            "diameter_km": 0.37,
            "albedo_pV": 0.23,
            "taxonomy": "Sq",
            "rot_per_h": 30.4,
            "a_AU": 0.9226,
            "ecc": 0.1914,
        },
    },
}

@dataclass
class HypatiaResult:
    """Resultado completo del pipeline maestro para un asteroide."""
    asteroid_id: int
    asteroid_name: str
    n_obs: Optional[int]
    t_years: float
    layer3_prior: dict
    layer2_posterior: object
    dadt_final: float
    dadt_std: float
    rmse_sin_yark: float
    rmse_hypatia: float
    reduccion_pct: float
    cone_width_final_km: float
    true_dadt: Optional[float] = None
    error_vs_true: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  HYPATIA — Resultado para {self.asteroid_name} ({self.asteroid_id})",
            "=" * 60,
            f"  N observaciones simuladas : {self.n_obs or 'completo'}",
            f"  Horizonte de predicción   : {self.t_years:.0f} años",
            " ",
            f"  da/dt posterior (Capa 2+3): {self.dadt_final:+.4f} ± {self.dadt_std:.4f} AU/My",
        ]
        if self.true_dadt is not None:
            lines.append(f"  da/dt real (JPL)          : {self.true_dadt:+.4f} AU/My")
            lines.append(f"  Error vs JPL              : {self.error_vs_true:.4f} AU/My  ({abs(self.error_vs_true/self.true_dadt)*100:.1f}%)")
        lines += [
            " ",
            f"  RMSE sin Yarkovsky        : {self.rmse_sin_yark:>10.0f} km",
            f"  RMSE HYPATIA              : {self.rmse_hypatia:>10.0f} km",
            f"  Reducción de error        : {self.reduccion_pct:>10.1f} %",
            " ",
            f"  Cono de incertidumbre     : {self.cone_width_final_km:>10.0f} km",
            "=" * 60,
        ]
        return "\n".join(lines)

def run_hypatia(
    asteroid_id: int = 99942,
    n_obs: Optional[int] = None,
    t_years: float = 40.0,
    model_path: Optional[str] = None,
    series_csv_path: Optional[str] = None,
    run_loocv: bool = False,
    save_results_path: Optional[str] = None,
    verbose: bool = True,
) -> HypatiaResult:
    """Pipeline maestro de HYPATIA: integra las tres capas."""
    cfg = ASTEROID_CONFIGS.get(asteroid_id)
    if cfg is None:
        raise ValueError(f"Asteroide {asteroid_id} no configurado. Disponibles: {list(ASTEROID_CONFIGS.keys())}")

    if verbose:
        print("\n" + "█" * 60)
        print(f"  HYPATIA — Pipeline maestro")
        print(f"  Asteroide: {cfg['name']} ({asteroid_id})")
        print(f"  N obs: {n_obs or 'completo'}  |  Horizonte: {t_years:.0f} años")
        print("█" * 60)

    # CAPA 3: ML → prior de da/dt
    if verbose: print("\n▶ CAPA 3: Inferencia ML del parámetro Yarkovsky")
    if model_path and Path(model_path).exists():
        l3 = run_layer3_offline(model_path, cfg["features"], verbose=verbose)
    else:
        l3 = run_layer3(
            asteroid_features=cfg["features"],
            model_path=model_path,
            run_loocv=run_loocv,
            run_benchmark=verbose,
            save_model_path=model_path,
            verbose=verbose,
        )
    ml_quantiles = l3.prior_quantiles

    # CAPA 2: Series de tiempo → posterior de da/dt
    if verbose: print("\n▶ CAPA 2: Estimación bayesiana desde residuos orbitales")
    if series_csv_path and Path(series_csv_path).exists():
        l2 = run_layer2_offline(
            series_csv_path=series_csv_path,
            ml_quantiles=ml_quantiles,
            n_obs_limit=n_obs,
            verbose=verbose,
        )
    else:
        l2 = run_layer2(
            asteroid_id=asteroid_id,
            epoch_start=cfg["epoch_valid"],
            epoch_end=cfg["epoch_end"],
            a_au=cfg["a_AU"],
            ecc=cfg["ecc"],
            ml_quantiles=ml_quantiles,
            n_obs_limit=n_obs,
            save_series_path=series_csv_path,
            verbose=verbose,
        )
    posterior = l2.posterior
    dadt_final = posterior.mean
    dadt_std = posterior.std

    # CAPA 1: EDOs → propagación con Yarkovsky + comparación
    if verbose: print("\n▶ CAPA 1: Propagación orbital y comparación de escenarios")
    ic = get_initial_conditions(asteroid_id, cfg["epoch_valid"], DEFAULT_PERTURBERS)
    y0, order, gm_map = pack_state_vector(ic)
    epoch_jd = ic["epoch_jd"]

    A2_true = dadt_to_A2(cfg["true_dadt"], cfg["a_AU"], cfg["ecc"])
    epoch_comp_end_jd = epoch_jd + t_years * 365.25
    
    try:
        ephemeris = fetch_ephemeris_arc(
            asteroid_id,
            jd_to_iso(epoch_jd),
            jd_to_iso(epoch_comp_end_jd),
            step="180d",
        )
    except Exception as e:
        if verbose:
            print(f"  [WARN] No se pudo descargar efemérides futuras: {e}")
            print("  [INFO] Usando efeméride con da/dt real como proxy de referencia")
        ref_result = propagate_from_state(y0, order, gm_map, t_years, A2_true, epoch_jd)
        ephemeris = {"times_jd": ref_result["times_jd"], "pos_au": ref_result["asteroid_pos"]}

    scenarios = {
        "sin_yark": 0.0,
        "hypatia": dadt_final,
        "jpl_ref": cfg["true_dadt"],
    }
    comparison = compare_scenarios(
        y0, order, gm_map, epoch_jd,
        ephemeris, scenarios,
        cfg["a_AU"], cfg["ecc"],
        t_years=t_years,
    )
    rmse_sin = comparison["sin_yark"]["rmse_km"]
    rmse_hyp = comparison["hypatia"]["rmse_km"]
    reduccion = comparison["hypatia"].get("reduction_pct", 0.0)

    # CONO DE INCERTIDUMBRE
    if verbose: print("\n▶ Generando cono de incertidumbre orbital...")
    cone = generate_uncertainty_cone(
        y0, order, gm_map, epoch_jd,
        dadt_mean=dadt_final,
        dadt_std=dadt_std,
        a_au=cfg["a_AU"],
        ecc=cfg["ecc"],
        t_years=t_years,
        n_samples=30,
    )
    cone_km = cone_width_at_year(cone, t_years)

    # RESULTADO FINAL
    error_vs_true = abs(dadt_final - cfg["true_dadt"]) if cfg.get("true_dadt") else None
    result = HypatiaResult(
        asteroid_id=asteroid_id,
        asteroid_name=cfg["name"],
        n_obs=n_obs,
        t_years=t_years,
        layer3_prior=ml_quantiles,
        layer2_posterior=posterior,
        dadt_final=dadt_final,
        dadt_std=dadt_std,
        rmse_sin_yark=rmse_sin,
        rmse_hypatia=rmse_hyp,
        reduccion_pct=reduccion,
        cone_width_final_km=cone_km,
        true_dadt=cfg.get("true_dadt"),
        error_vs_true=error_vs_true,
    )
    if verbose: print(result.summary())
    if save_results_path:
        _save_result_json(result, save_results_path)
    return result

def run_sensitivity_experiment(
    asteroid_id: int = 99942,
    n_obs_list: list = [5, 10, 20, 50],
    t_years: float = 40.0,
    model_path: Optional[str] = None,
    series_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Experimento central: RMSE vs N observaciones."""
    rows = []
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  EXPERIMENTO CENTRAL: RMSE vs N observaciones")
        print(f"  Asteroide: {asteroid_id}  |  Horizonte: {t_years:.0f} años")
        print(f"{'═'*60}")
    for n in n_obs_list:
        if verbose: print(f"\n  → N = {n} observaciones...")
        try:
            res = run_hypatia(
                asteroid_id=asteroid_id,
                n_obs=n,
                t_years=t_years,
                model_path=model_path,
                series_csv_path=series_path,
                run_loocv=False,
                verbose=False,
            )
            rows.append({
                "n_obs": n,
                "dadt_posterior": res.dadt_final,
                "dadt_std": res.dadt_std,
                "rmse_sin_yark": res.rmse_sin_yark,
                "rmse_hypatia": res.rmse_hypatia,
                "reduccion_pct": res.reduccion_pct,
                "cone_km": res.cone_width_final_km,
            })
            if verbose:
                print(f"    da/dt = {res.dadt_final:+.4f} AU/My  |  Reducción = {res.reduccion_pct:.1f}%")
        except Exception as e:
            print(f"  [ERROR] N={n}: {e}")
    df = pd.DataFrame(rows)
    if verbose and len(df) > 0:
        print(f"\n{'═'*60}")
        print("  RESUMEN DEL EXPERIMENTO:")
        print(df.to_string(index=False, float_format="{:.3f}".format))
        print(f"{'═'*60}")
    return df

def _save_result_json(result: HypatiaResult, path: str) -> None:
    """Guarda el resultado en JSON para reproducibilidad."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "asteroid_id": result.asteroid_id,
        "asteroid_name": result.asteroid_name,
        "n_obs": result.n_obs,
        "t_years": result.t_years,
        "dadt_final": result.dadt_final,
        "dadt_std": result.dadt_std,
        "rmse_sin_yark_km": result.rmse_sin_yark,
        "rmse_hypatia_km": result.rmse_hypatia,
        "reduccion_pct": result.reduccion_pct,
        "cone_width_final_km": result.cone_width_final_km,
        "true_dadt": result.true_dadt,
        "error_vs_true": result.error_vs_true,
        "layer3_prior": {str(k): v for k, v in result.layer3_prior.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[HYPATIA] Resultado guardado: {path}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYPATIA — Pipeline maestro de predicción orbital")
    parser.add_argument("--target", type=int, default=99942, help="ID JPL del asteroide (default: 99942 = Apophis)")
    parser.add_argument("--n-obs", type=int, default=None, help="Simular N observaciones iniciales (default: completo)")
    parser.add_argument("--years", type=float, default=40.0, help="Horizonte de predicción en años (default: 40)")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo ML serializado (.joblib)")
    parser.add_argument("--series", type=str, default=None, help="Ruta al CSV de residuos orbitales")
    parser.add_argument("--loocv", action="store_true", help="Ejecutar LOO-CV completo (lento)")
    parser.add_argument("--experiment", action="store_true", help="Correr experimento completo de sensibilidad")
    parser.add_argument("--save", type=str, default=None, help="Guardar resultado en JSON")
    args = parser.parse_args()

    if args.experiment:
        df = run_sensitivity_experiment(
            asteroid_id=args.target,
            n_obs_list=[5, 10, 20, 50],
            t_years=args.years,
            model_path=args.model,
            series_path=args.series,
        )
        if args.save:
            df.to_csv(args.save.replace(".json", ".csv"), index=False)
    else:
        result = run_hypatia(
            asteroid_id=args.target,
            n_obs=args.n_obs,
            t_years=args.years,
            model_path=args.model,
            series_csv_path=args.series,
            run_loocv=args.loocv,
            save_results_path=args.save,
        )