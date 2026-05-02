"""
tests/test_layer1_ode.py
------------------------
Tests unitarios para la Capa 1 de HYPATIA.

Ejecutar con:
    pytest tests/test_layer1_ode.py -v

Los tests marcados con @pytest.mark.slow requieren conexión a JPL Horizons
y tardan varios minutos. Correrlos explícitamente:
    pytest tests/test_layer1_ode.py -v -m slow
"""

import pytest
import numpy as np

from src.layer1_ode.yarkovsky import (
    yarkovsky_acceleration,
    dadt_to_A2,
    A2_to_dadt,
    yarkovsky_order_of_magnitude,
)
from src.layer1_ode.utils import (
    state_to_orbital_elements,
    semi_major_axis,
    check_energy_conservation,
    au_to_km,
    jd_to_iso,
    iso_to_jd,
)
from src.layer1_ode.constants import GM_SOL_AU3_DAY2


# ── YARKOVSKY ─────────────────────────────────────────────────────────────

class TestYarkovskyAcceleration:

    def test_zero_A2_returns_zero(self):
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.array([0.0, 0.01720, 0.0])
        acc = yarkovsky_acceleration(pos, vel, A2=0.0)
        np.testing.assert_array_equal(acc, np.zeros(3))

    def test_direction_tangential(self):
        """La aceleración debe ser paralela a la velocidad."""
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.array([0.0, 0.01720, 0.0])   # velocidad circular aprox.
        A2  = 1e-14
        acc = yarkovsky_acceleration(pos, vel, A2=A2)
        # acc debe ser paralelo a vel
        cross = np.cross(acc, vel)
        assert np.linalg.norm(cross) < 1e-20

    def test_sign_convention(self):
        """A2 positivo → empuje en dirección del movimiento (da/dt > 0)."""
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.array([0.0, 0.01720, 0.0])
        acc_pos = yarkovsky_acceleration(pos, vel, A2=+1e-14)
        acc_neg = yarkovsky_acceleration(pos, vel, A2=-1e-14)
        assert acc_pos[1] > 0   # mismo sentido que vel_y
        assert acc_neg[1] < 0   # sentido opuesto

    def test_distance_scaling(self):
        """La aceleración debe caer como 1/r² al alejarse del Sol."""
        vel = np.array([0.0, 0.01720, 0.0])
        A2  = 1e-14
        acc_1au = yarkovsky_acceleration(np.array([1.0, 0.0, 0.0]), vel, A2)
        acc_2au = yarkovsky_acceleration(np.array([2.0, 0.0, 0.0]), vel, A2)
        ratio = np.linalg.norm(acc_1au) / np.linalg.norm(acc_2au)
        assert abs(ratio - 4.0) < 0.01   # (2/1)² = 4

    def test_zero_velocity_returns_zero(self):
        """Velocidad nula no debe causar división por cero."""
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.zeros(3)
        acc = yarkovsky_acceleration(pos, vel, A2=1e-14)
        np.testing.assert_array_equal(acc, np.zeros(3))


class TestDadtConversion:

    def test_roundtrip_dadt_A2(self):
        """dadt → A2 → dadt debe recuperar el valor original."""
        dadt_original = -0.20   # AU/My (valor de Apophis)
        a_au = 0.9226
        ecc  = 0.1914
        A2   = dadt_to_A2(dadt_original, a_au, ecc)
        dadt_recovered = A2_to_dadt(A2, a_au, ecc)
        assert abs(dadt_recovered - dadt_original) < 1e-8

    def test_apophis_A2_magnitude(self):
        """A2 de Apophis debe estar en el orden 1e-14 AU/día²."""
        A2 = dadt_to_A2(-0.20, a_au=0.9226, ecc=0.1914)
        assert 1e-13 < abs(A2) < 1e-11

    def test_dadt_zero_gives_A2_zero(self):
        A2 = dadt_to_A2(0.0, a_au=1.0, ecc=0.0)
        assert A2 == 0.0

    def test_order_of_magnitude_apophis(self):
        """Estimación de |da/dt| para Apophis (~370m) debe ser ~0.1-0.5 AU/My."""
        mag = yarkovsky_order_of_magnitude(diameter_km=0.37, a_au=0.9226)
        assert 0.05 < abs(mag) < 1.0


# ── ELEMENTOS ORBITALES ───────────────────────────────────────────────────

class TestOrbitalElements:

    def _earth_state(self):
        """Estado aproximado de la Tierra en J2000 (1 AU, órbita circular)."""
        v_circ = np.sqrt(GM_SOL_AU3_DAY2 / 1.0)   # velocidad circular en 1 AU
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, v_circ, 0.0])

    def test_earth_semi_major_axis(self):
        """La Tierra debe tener a ≈ 1 AU."""
        pos, vel = self._earth_state()
        elems = state_to_orbital_elements(pos, vel)
        assert abs(elems["a"] - 1.0) < 0.01

    def test_earth_eccentricity(self):
        """Órbita circular debe tener e ≈ 0."""
        pos, vel = self._earth_state()
        elems = state_to_orbital_elements(pos, vel)
        assert elems["e"] < 0.01

    def test_semi_major_axis_consistency(self):
        """semi_major_axis() debe coincidir con state_to_orbital_elements()."""
        pos, vel = self._earth_state()
        a_direct = semi_major_axis(pos, vel)
        a_full   = state_to_orbital_elements(pos, vel)["a"]
        assert abs(a_direct - a_full) < 1e-10


# ── UTILIDADES ────────────────────────────────────────────────────────────

class TestUtils:

    def test_au_to_km(self):
        assert abs(au_to_km(1.0) - 1.495978707e8) < 1e3

    def test_jd_iso_roundtrip(self):
        iso_original = "2029-04-13"
        jd = iso_to_jd(iso_original)
        iso_back = jd_to_iso(jd)
        assert iso_back == iso_original

    def test_jd_apophis_approach(self):
        """El acercamiento de Apophis es el 13 de abril de 2029."""
        jd = iso_to_jd("2029-04-13")
        assert 2462000 < jd < 2463000   # rango esperado


# ── TESTS LENTOS (requieren JPL Horizons) ────────────────────────────────

@pytest.mark.slow
class TestIntegrationWithJPL:

    def test_propagation_runs(self):
        """El integrador debe completar sin errores para Apophis."""
        from src.layer1_ode import propagate
        result = propagate(
            asteroid_id=99942,
            epoch_start="2024-01-01",
            t_years=1.0,
            dadt_au_my=0.0,
            a_au=0.9226,
            ecc=0.1914,
        )
        assert result["asteroid_pos"].shape[1] == 3
        assert len(result["times_jd"]) > 0

    def test_energy_conservation(self):
        """La energía orbital debe conservarse con variación < 1e-6."""
        from src.layer1_ode import propagate
        from src.layer1_ode.utils import check_energy_conservation
        result = propagate(99942, "2024-01-01", t_years=5.0)
        ec = check_energy_conservation(result)
        assert ec["passed"], (
            f"Energía no conservada: variación relativa = {ec['variation_rel']:.2e}"
        )

    def test_validation_passes(self):
        """RMSE de validación debe ser < 1000 km en arco de 10 años."""
        from src.layer1_ode import run_validation
        report = run_validation(
            asteroid_id=99942,
            epoch_start="2014-01-01",
            arc_years=10.0,
        )
        assert report["passed"], (
            f"Validación fallida: RMSE = {report['rmse_km']:.0f} km"
        )
