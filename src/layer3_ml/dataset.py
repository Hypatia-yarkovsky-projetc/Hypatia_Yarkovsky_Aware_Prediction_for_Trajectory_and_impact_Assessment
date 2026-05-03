"""
dataset.py
Construcción del dataset de entrenamiento para el modelo ML de HYPATIA.
Fuentes: JPL SBDB (A2), WISE/NEOWISE (albedo/diámetro), LCDB (rotación).
Estrategia: imputación física justificada y conversión A2 → da/dt.
"""
import numpy as np
import pandas as pd
import requests
import json
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

JPL_SBDB_URL  = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
PROCESSED_DIR = Path("data/processed")

TAX_ALBEDO_MEDIAN = {
    "A": 0.23, "B": 0.07, "C": 0.06, "D": 0.05,
    "E": 0.45, "F": 0.05, "G": 0.11, "K": 0.14,
    "L": 0.16, "M": 0.17, "O": 0.26, "Q": 0.20,
    "R": 0.35, "S": 0.23, "T": 0.07, "V": 0.35,
    "X": 0.15, "Xe": 0.15, "Xc": 0.09, "Xk": 0.13,
    "Sq": 0.22, "Sr": 0.25, "Sa": 0.22, "Sl": 0.18,
    "Cb": 0.06, "Ch": 0.07, "Cg": 0.06,
}

TAX_CODE = {
    "B": 0.05, "C": 0.07, "Cb": 0.07, "Cg": 0.07, "Ch": 0.07, "F": 0.08,
    "G": 0.09, "D": 0.10, "T": 0.11, "X": 0.40, "Xc": 0.38, "Xe": 0.45,
    "Xk": 0.42, "K": 0.50, "L": 0.52, "Sl": 0.55, "Sa": 0.58, "Sq": 0.60,
    "Sr": 0.63, "S": 0.65, "Q": 0.68, "O": 0.72, "R": 0.75, "A": 0.80,
    "V": 0.85, "E": 0.90, "M": 0.45,
}

def download_yarkovsky_sbdb(
    save_path: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Descarga asteroides con parámetro Yarkovsky (A2) medido desde JPL SBDB."""
    print("[HYPATIA L3] Descargando dataset Yarkovsky desde JPL SBDB...")

    fields = (
        "spkid,full_name,neo,pha,class,H,albedo,diameter,"
        "rot_per,spec_T,spec_B,a,e,w,i,om,ma,"
        "A2,dA2"
    )

    params = {
        "fields"   : fields,
        "sb-kind"  : "a",
        "sb-cdata" : json.dumps({"AND": [{"field": "A2", "value": "0", "op": "ne"}]}),
        "limit"    : 600,
    }

    try:
        resp = requests.get(JPL_SBDB_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[HYPATIA L3] Error de red: {e}")
        print("[HYPATIA L3] Cargando dataset de respaldo embebido...")
        return _get_fallback_dataset()

    if "data" not in data or len(data["data"]) == 0:
        print("[HYPATIA L3] Sin resultados. Usando dataset de respaldo.")
        return _get_fallback_dataset()

    col_names = [f["name"] for f in data["fields"]]
    df = pd.DataFrame(data["data"], columns=col_names)

    numeric_cols = ["H", "albedo", "diameter", "rot_per", "a", "e", "w",
                    "i", "om", "ma", "A2", "dA2"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[HYPATIA L3] Descargados {len(df)} asteroides con A2 medido.")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[HYPATIA L3] Guardado: {save_path}")

    return df

def _get_fallback_dataset() -> pd.DataFrame:
    """Dataset de respaldo con asteroides conocidos. Usado cuando la API falla."""
    records = [
        {"spkid": 2099942, "full_name": "99942 Apophis", "class": "Aten", "H": 19.2, "albedo": 0.23, "diameter": 0.37,
         "rot_per": 30.4, "spec_T": "Sq", "a": 0.9226, "e": 0.1914, "i": 3.34, "A2": -4.53e-14, "dA2": 6.81e-15},
        {"spkid": 2101955, "full_name": "101955 Bennu", "class": "Apollo", "H": 20.2, "albedo": 0.044, "diameter": 0.49,
         "rot_per": 4.3, "spec_T": "B", "a": 1.1264, "e": 0.2037, "i": 6.03, "A2": -19.0e-14, "dA2": 0.1e-14},
        {"spkid": 2162173, "full_name": "162173 Ryugu", "class": "Apollo", "H": 18.8, "albedo": 0.045, "diameter": 0.90,
         "rot_per": 7.63, "spec_T": "C", "a": 1.1896, "e": 0.1902, "i": 5.88, "A2": -5.1e-14, "dA2": 1.2e-14},
        {"spkid": 2101956, "full_name": "1950 DA", "class": "Apollo", "H": 17.0, "albedo": 0.14, "diameter": 1.30,
         "rot_per": 2.12, "spec_T": "E", "a": 1.6997, "e": 0.5077, "i": 12.17, "A2": 1.77e-14, "dA2": 0.56e-14},
        {"spkid": 2001620, "full_name": "1620 Geographos", "class": "Apollo", "H": 15.5, "albedo": 0.33, "diameter": 2.00,
         "rot_per": 5.22, "spec_T": "S", "a": 1.2454, "e": 0.3354, "i": 13.34, "A2": 1.15e-15, "dA2": 0.40e-15},
        {"spkid": 2000433, "full_name": "433 Eros", "class": "Amor", "H": 11.2, "albedo": 0.25, "diameter": 16.84,
         "rot_per": 5.27, "spec_T": "S", "a": 1.4582, "e": 0.2228, "i": 10.83, "A2": 1.5e-15, "dA2": 0.4e-15},
        {"spkid": 2025143, "full_name": "25143 Itokawa", "class": "Apollo", "H": 18.9, "albedo": 0.53, "diameter": 0.32,
         "rot_per": 12.13, "spec_T": "S", "a": 1.3241, "e": 0.2801, "i": 1.62, "A2": -6.3e-14, "dA2": 1.6e-14},
        {"spkid": 2002100, "full_name": "2100 Ra-Shalom", "class": "Aten", "H": 16.1, "albedo": 0.08, "diameter": 2.30,
         "rot_per": 19.8, "spec_T": "C", "a": 0.8320, "e": 0.4365, "i": 15.76, "A2": -3.2e-15, "dA2": 1.1e-15},
        {"spkid": 2003200, "full_name": "3200 Phaethon", "class": "Apollo", "H": 14.3, "albedo": 0.11, "diameter": 5.10,
         "rot_per": 3.60, "spec_T": "B", "a": 1.2712, "e": 0.8898, "i": 22.26, "A2": -5.0e-15, "dA2": 1.5e-15},
        {"spkid": 2002062, "full_name": "2062 Aten", "class": "Aten", "H": 17.0, "albedo": 0.20, "diameter": 0.90,
         "rot_per": 40.77, "spec_T": "S", "a": 0.9669, "e": 0.1826, "i": 18.93, "A2": -1.4e-14, "dA2": 0.5e-14},
        {"spkid": 2009969, "full_name": "9969 Braille", "class": "Apollo", "H": 15.6, "albedo": 0.34, "diameter": 2.10,
         "rot_per": 226.4, "spec_T": "Q", "a": 2.3436, "e": 0.4347, "i": 28.97, "A2": 2.8e-15, "dA2": 0.8e-15},
        {"spkid": 3162660, "full_name": "2005 QQ87", "class": "Apollo", "H": 20.5, "albedo": 0.25, "diameter": 0.30,
         "rot_per": 2.80, "spec_T": "S", "a": 1.0720, "e": 0.2141, "i": 6.74, "A2": -5.2e-14, "dA2": 2.1e-14},
        {"spkid": 2001685, "full_name": "1685 Toro", "class": "Apollo", "H": 14.2, "albedo": 0.22, "diameter": 3.60,
         "rot_per": 10.20, "spec_T": "S", "a": 1.3676, "e": 0.4361, "i": 9.38, "A2": 1.9e-15, "dA2": 0.7e-15},
        {"spkid": 2001566, "full_name": "1566 Icarus", "class": "Apollo", "H": 16.4, "albedo": 0.51, "diameter": 1.40,
         "rot_per": 2.27, "spec_T": "S", "a": 1.0778, "e": 0.8268, "i": 22.85, "A2": 4.5e-15, "dA2": 1.8e-15},
        {"spkid": 2002063, "full_name": "2063 Bacchus", "class": "Apollo", "H": 17.1, "albedo": 0.19, "diameter": 1.10,
         "rot_per": 14.90, "spec_T": "Sq", "a": 1.0781, "e": 0.3495, "i": 9.43, "A2": -2.1e-14, "dA2": 0.8e-14},
        {"spkid": 2004660, "full_name": "4660 Nereus", "class": "Apollo", "H": 18.2, "albedo": 0.55, "diameter": 0.51,
         "rot_per": 15.16, "spec_T": "Xe", "a": 1.4888, "e": 0.3600, "i": 1.43, "A2": -2.4e-14, "dA2": 0.7e-14},
        {"spkid": 2002340, "full_name": "2340 Hathor", "class": "Aten", "H": 20.0, "albedo": 0.05, "diameter": 0.29,
         "rot_per": 3.35, "spec_T": "C", "a": 0.8441, "e": 0.4499, "i": 5.85, "A2": -7.8e-14, "dA2": 3.2e-14},
        {"spkid": 2003361, "full_name": "3361 Orpheus", "class": "Apollo", "H": 19.0, "albedo": 0.32, "diameter": 0.35,
         "rot_per": 3.53, "spec_T": "S", "a": 1.2094, "e": 0.3229, "i": 2.68, "A2": -3.9e-14, "dA2": 1.6e-14},
        {"spkid": 2085774, "full_name": "85774 (1998 UT18)", "class": "Aten", "H": 19.3, "albedo": 0.09, "diameter": 0.40,
         "rot_per": 8.40, "spec_T": "C", "a": 0.8691, "e": 0.3541, "i": 5.37, "A2": -6.1e-14, "dA2": 2.5e-14},
        {"spkid": 3057791, "full_name": "2000 SG344", "class": "Apollo", "H": 24.7, "albedo": 0.30, "diameter": 0.04,
         "rot_per": np.nan, "spec_T": "S", "a": 0.9776, "e": 0.0669, "i": 0.11, "A2": -4.8e-13, "dA2": 2.0e-13},
    ]
    df = pd.DataFrame(records)
    df["dA2"] = df["dA2"].fillna(df["A2"].abs() * 0.3)
    print(f"[HYPATIA L3] Dataset de respaldo cargado: {len(df)} asteroides")
    return df

def compute_dadt_from_A2(
    df: pd.DataFrame,
    gm_sun: float = 0.01720209895**2,
) -> pd.DataFrame:
    """Convierte el parámetro A2 (AU/día²) a da/dt (AU/My)."""
    df = df.copy()

    a = pd.to_numeric(df["a"], errors="coerce").fillna(1.0)
    e = pd.to_numeric(df["e"], errors="coerce").fillna(0.2).clip(0, 0.99)
    A2 = pd.to_numeric(df["A2"], errors="coerce")

    n = np.sqrt(gm_sun / a**3)
    factor = np.sqrt(np.maximum(1 - e**2, 1e-6))

    dadt_au_day = 2.0 * A2 / (n * a * factor)
    df["dadt_AuMy"] = dadt_au_day / (1.0 / (365.25e6))

    if "dA2" in df.columns:
        dA2 = pd.to_numeric(df["dA2"], errors="coerce").fillna(A2.abs() * 0.2)
        dadt_sig = 2.0 * dA2 / (n * a * factor) / (1.0 / (365.25e6))
        df["dadt_sig_AuMy"] = dadt_sig.abs()
    else:
        df["dadt_sig_AuMy"] = df["dadt_AuMy"].abs() * 0.2

    return df

def build_training_dataset(
    sbdb_path     : Optional[str] = None,
    save_path     : Optional[str] = None,
    verbose       : bool = True,
) -> pd.DataFrame:
    """Construye el dataset de entrenamiento final listo para el modelo ML."""
    if sbdb_path and Path(sbdb_path).exists():
        df = pd.read_csv(sbdb_path)
        if verbose:
            print(f"[HYPATIA L3] Datos cargados desde caché: {sbdb_path}")
    else:
        df = download_yarkovsky_sbdb(save_path=sbdb_path)

    col_map = {
        "spec_T": "taxonomy", "spec_B": "taxonomy_B",
        "rot_per": "rot_per_h", "albedo": "albedo_pV",
        "diameter": "diameter_km", "dA2": "A2_sig",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "taxonomy_B" in df.columns:
        df["taxonomy"] = df["taxonomy"].fillna(df["taxonomy_B"])

    if "A2" in df.columns:
        df = compute_dadt_from_A2(df)
    else:
        raise ValueError("Columna 'A2' requerida en el dataset.")

    df["taxonomy"] = df["taxonomy"].fillna("S").str.strip().str.split().str[0]

    df["albedo_pV"] = pd.to_numeric(
        df.get("albedo_pV", pd.Series(dtype=float)), errors="coerce"
    )
    tax_series = df["taxonomy"].str[0].str.upper()
    df["albedo_pV"] = df["albedo_pV"].fillna(
        tax_series.map(TAX_ALBEDO_MEDIAN).fillna(0.15)
    )

    df["diameter_km"] = pd.to_numeric(
        df.get("diameter_km", pd.Series(dtype=float)), errors="coerce"
    )
    H = pd.to_numeric(df.get("H", pd.Series(dtype=float)), errors="coerce")
    pV = df["albedo_pV"].clip(0.01, 1.0)
    diam_from_H = 1329.0 / np.sqrt(pV) * 10**(-H / 5.0)
    df["diameter_km"] = df["diameter_km"].fillna(diam_from_H).clip(lower=0.001)

    df["rot_per_h"] = pd.to_numeric(
        df.get("rot_per_h", pd.Series(dtype=float)), errors="coerce"
    )
    df["rot_per_h"] = df["rot_per_h"].fillna(7.0).clip(lower=0.1)

    for col in ["a", "e", "i"]:
        df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)),
                                errors="coerce")
    df["a"] = df["a"].fillna(1.2).clip(lower=0.1)
    df["e"] = df["e"].fillna(0.2).clip(0.0, 0.99)
    df["i"] = df["i"].fillna(10.0).clip(0.0, 90.0)

    df["tax_code"]     = df["taxonomy"].str[0].str.upper().map(TAX_CODE).fillna(0.5)
    df["inv_diameter"] = 1.0 / df["diameter_km"]
    df["absorptivity"] = 1.0 - df["albedo_pV"].clip(0, 1)

    valid = (
        df["dadt_AuMy"].notna()
        & df["dadt_AuMy"].abs().between(1e-5, 10.0)
        & df["diameter_km"].gt(0)
        & df["a"].gt(0)
    )
    df = df[valid].reset_index(drop=True)

    final_cols = [
        "spkid", "full_name", "taxonomy", "tax_code",
        "diameter_km", "inv_diameter",
        "albedo_pV", "absorptivity",
        "rot_per_h", "a", "e", "i",
        "dadt_AuMy", "dadt_sig_AuMy",
    ]
    available = [c for c in final_cols if c in df.columns]
    df = df[available].copy()
    df = df.rename(columns={"a": "a_AU", "e": "ecc", "i": "incl_deg"})

    if verbose:
        print(f"\n[HYPATIA L3] Dataset de entrenamiento construido:")
        print(f"  Filas          : {len(df)}")
        print(f"  da/dt rango    : [{df['dadt_AuMy'].min():.4f}, {df['dadt_AuMy'].max():.4f}] AU/My")
        print(f"  da/dt media    : {df['dadt_AuMy'].mean():.4f} AU/My")
        print(f"  Diámetro rango : [{df['diameter_km'].min():.3f}, {df['diameter_km'].max():.1f}] km")
        t_counts = df["taxonomy"].str[0].value_counts()
        print(f"  Taxonomía (top): {dict(t_counts.head(5))}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[HYPATIA L3] Dataset guardado: {save_path}")

    return df

def load_training_dataset(path: str) -> pd.DataFrame:
    """Carga dataset de entrenamiento desde CSV procesado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {path}\n"
            "Ejecuta build_training_dataset() primero."
        )
    df = pd.read_csv(path)
    required = ["dadt_AuMy", "inv_diameter", "absorptivity", "tax_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset mal formado, columnas faltantes: {missing}")
    return df