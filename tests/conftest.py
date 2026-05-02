"""
tests/conftest.py
-----------------
Configuración global de pytest para HYPATIA.

Marca los tests lentos (requieren JPL Horizons) para poder
excluirlos en ejecuciones rápidas:

    pytest tests/ -v              # corre solo tests rápidos
    pytest tests/ -v -m slow      # corre SOLO los lentos
    pytest tests/ -v --all        # corre todos
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--all", action="store_true", default=False,
        help="Incluir tests lentos que requieren conexión a JPL Horizons"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests que requieren conexión a JPL Horizons (tardan varios minutos)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--all"):
        skip_slow = pytest.mark.skip(
            reason="Test lento omitido. Usa --all para incluirlo."
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
