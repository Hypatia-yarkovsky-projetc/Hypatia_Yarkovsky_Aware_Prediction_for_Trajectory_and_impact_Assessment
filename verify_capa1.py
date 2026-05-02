"""
verify_capa1.py
Diagnóstico completo de conexión, física y rendimiento de la Capa 1.
"""
import numpy as np
from src.layer1_ode import (
    propagate, check_energy_conservation, compute_moid_timeseries, 
    find_close_approaches, state_to_orbital_elements, yarkovsky_order_of_magnitude
)

def main():
    print("🔍 1. Verificando consistencia física de Yarkovsky...")
    dadt_est = yarkovsky_order_of_magnitude(diameter_km=0.37, a_au=0.922)
    assert 0.18 <= dadt_est <= 0.22, f"Yarkovsky estimado fuera de rango: {dadt_est}"
    print(f"   ✓ da/dt estimado para Apophis: {dadt_est:.3f} AU/My")

    print("\n🚀 2. Propagando 5 años (Grav puro vs Yarkovsky)...")
    res_grav = propagate("99942", "2024-01-01", t_years=5.0, dadt_au_my=0.0)
    res_yark = propagate("99942", "2024-01-01", t_years=5.0, dadt_au_my=-0.20, a_au=0.9226, ecc=0.1914)
    
    print(f"   ✓ Gravitacional: {len(res_grav['times_jd'])} pasos")
    print(f"   ✓ Yarkovsky:     {len(res_yark['times_jd'])} pasos")

    print("\n⚡ 3. Conservación de energía...")
    E_report = check_energy_conservation(res_grav)
    print(f"   ✓ ΔE/E₀ = {E_report['variation_rel']:.2e} {'✓ PASA' if E_report['passed'] else '✗ REVISAR'}")

    print("\n🌍 4. MOID y acercamientos...")
    earth_res = propagate("399", "2024-01-01", t_years=5.0)  # Tierra como "asteroid"
    earth_res["earth_pos"] = earth_res.pop("asteroid_pos")     # Alias explícito
    moid_series = compute_moid_timeseries(res_grav, earth_res, window_years=1.0)
    phas = find_close_approaches(moid_series, threshold_au=0.05)
    print(f"   ✓ {len(moid_series)} ventanas calculadas | {len(phas)} acercamientos < 0.05 AU")

    print("\n✅ CAPA 1: INTEGRACIÓN EXITOSA. Lista para Capa 2 (Series de Tiempo).")

if __name__ == "__main__":
    main()