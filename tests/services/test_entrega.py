"""Verificación previa a la entrega. Ejecutar con:  python -m pytest -m entrega

Falla mientras exista el backend provisional o falten los módulos reales.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.services import PredictionService, ServiceConfig
from src.services.config import DEFAULT_MODEL, DEFAULT_PROCESSING, PROJECT_ROOT

pytestmark = pytest.mark.entrega


def test_modulos_reales_presentes():
    def _find(m):
        try:
            return importlib.util.find_spec(m) is not None
        except ModuleNotFoundError:
            return False

    faltan = [m for m in (DEFAULT_PROCESSING, DEFAULT_MODEL) if not _find(m)]
    assert not faltan, f"faltan módulos reales: {faltan}"


def test_backend_provisional_eliminado():
    assert not (PROJECT_ROOT / "src" / "services" / "provisional").exists(), (
        "elimine src/services/provisional/ antes de entregar"
    )


def test_servicio_real_arranca():
    svc = PredictionService(ServiceConfig(backend="real"))
    assert svc.info.backend == "real"


def test_predicciones_de_entrega_sin_simulacion(tmp_path):
    """Con el backend real, el lote de ejemplo completo se exporta en formato oficial."""
    import pandas as pd
    from src.services.config import DEFAULT_CASES

    svc = PredictionService(ServiceConfig(backend="real"))
    casos = pd.read_csv(DEFAULT_CASES, dtype=str, keep_default_na=False)
    out = svc.export(svc.predict_batch(casos), tmp_path / "predicciones.csv", expected_case_ids=casos["caso_id"])
    assert out.read_text(encoding="utf-8").splitlines()[0] == "caso_id,rango,ruta"
