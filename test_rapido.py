"""test_rapido.py → Prueba de integración corta (1 año) para desarrollo iterativo."""
import time
from src.layer1_ode import propagate

def main():
    print(" Iniciando prueba rápida (1 año: 2024 → 2025)...")
    t0 = time.time()
    res = propagate("99942", "2024-01-01", t_years=1.0, dadt_au_my=0.0)
    elapsed = time.time() - t0
    
    print(f" ¡INTEGRACIÓN EXITOSA en {elapsed:.2f} segundos!")
    print(f"   Posición final (X, Y, Z): {res['asteroid_pos'][-1]} AU")
    print(f"   Pasos de integración: {len(res['times_jd'])}")

if __name__ == "__main__":
    main()