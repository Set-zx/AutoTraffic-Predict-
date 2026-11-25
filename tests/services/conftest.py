from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.services import ServiceConfig, reset_service
from src.services.config import DEFAULT_MODEL, DEFAULT_PROCESSING, EVALUATION_DIR, RAW_DIR

DATA_DIR = RAW_DIR
CASES_CSV = EVALUATION_DIR / "casos_ejemplo.csv"
DATA_FILES = ("nodos.csv", "tramos.csv", "vehiculos.csv", "captaciones.csv")
REAL_MODULES = (DEFAULT_PROCESSING, DEFAULT_MODEL)


REQUIRED = {DEFAULT_PROCESSING: ("load_network", "load_trips"), DEFAULT_MODEL: ("build_model",)}


def _implements_contract(mod: str) -> bool:
    """True solo si el módulo se importa Y expone las funciones del contrato.

    Una carpeta con archivos vacíos (andamiaje del repositorio) no cuenta como módulo real.
    """
    try:
        m = importlib.import_module(mod)
    except Exception:
        return False
    return all(callable(getattr(m, fn, None)) for fn in REQUIRED[mod])


def _real_modules_available() -> bool:
    return all(_implements_contract(m) for m in REAL_MODULES)


# --- Marcador "entrega" sin depender de archivos compartidos (pyproject.toml) ---
def pytest_configure(config):
    config.addinivalue_line("markers", "entrega: verificación previa a la entrega (pytest -m entrega)")


def pytest_collection_modifyitems(config, items):
    """Las pruebas de entrega solo corren si se piden explícitamente con -m entrega."""
    if "entrega" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="solo con: python -m pytest -m entrega")
    for item in items:
        if "entrega" in item.keywords:
            item.add_marker(skip)


# Si los módulos de Daniel y Omar ya están en el repo, las pruebas de integración los usan;
# si no, usan el backend provisional. Tras la integración, el provisional se elimina.
BACKEND = "real" if _real_modules_available() else "provisional"


def tramos_graph() -> nx.Graph:
    """Red construida directamente de tramos.csv (referencia independiente para las pruebas)."""
    t = pd.read_csv(DATA_DIR / "tramos.csv")
    g = nx.Graph()
    g.add_edges_from(zip(t["origen"], t["destino"]))
    return g


@pytest.fixture(autouse=True)
def _fresh_service():
    reset_service()
    yield
    reset_service()


@pytest.fixture(scope="session")
def data_dir_sin_soluciones(tmp_path_factory) -> Path:
    """Copia de data/raw SIN ningún archivo de soluciones: prueba que el servicio no lo necesita."""
    d = tmp_path_factory.mktemp("datos")
    for f in DATA_FILES:
        shutil.copy(DATA_DIR / f, d / f)
    return d


@pytest.fixture
def svc_config(data_dir_sin_soluciones) -> ServiceConfig:
    return ServiceConfig(data_dir=data_dir_sin_soluciones, backend=BACKEND)


@pytest.fixture(scope="session")
def casos_ejemplo() -> pd.DataFrame:
    return pd.read_csv(CASES_CSV, dtype=str, keep_default_na=False)


@pytest.fixture
def caso_valido() -> dict:
    return {
        "caso_id": "T001",
        "vehiculo_id": "VF-11",
        "punto_inicial": "P3",
        "fecha_hora_inicio": "2026-09-25 08:05:00",
    }
