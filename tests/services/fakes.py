"""Dobles de prueba (solo para tests). Permiten provocar salidas inválidas del modelo
o datos que rompen el contrato. No forman parte de la aplicación: nunca se importan
desde src/ y no producen predicciones entregables.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from src.services import ServiceConfig


class ScriptedModel:
    def __init__(self, salida):
        self.salida = salida
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, trips):
        self.fit_calls += 1
        return self

    def predict(self, vehiculo_id, punto_inicial, fecha_hora_inicio, k=3):
        self.predict_calls += 1
        if isinstance(self.salida, Exception):
            raise self.salida
        return self.salida(punto_inicial) if callable(self.salida) else self.salida


def model_module(salida) -> SimpleNamespace:
    modelo = ScriptedModel(salida)
    return SimpleNamespace(__name__="tests.fake_model", build_model=lambda graph: modelo, instancia=modelo)


def processing_con_trips(config: ServiceConfig, transform) -> SimpleNamespace:
    """Envuelve el procesamiento configurado y altera load_trips para romper el contrato."""
    base = importlib.import_module(config.resolved_processing_module)
    return SimpleNamespace(
        __name__="tests.fake_processing",
        load_network=base.load_network,
        load_trips=lambda d, s, e: transform(base.load_trips(d, s, e), d),
    )


def counting_model_module(config: ServiceConfig) -> SimpleNamespace:
    """Envuelve el módulo de modelo configurado para contar cuántas veces se entrena."""
    base = importlib.import_module(config.resolved_model_module)
    contador = SimpleNamespace(builds=0, fits=0)

    def build_model(graph):
        contador.builds += 1
        modelo = base.build_model(graph)
        fit_original = modelo.fit

        def fit(trips):
            contador.fits += 1
            return fit_original(trips)

        modelo.fit = fit
        return modelo

    return SimpleNamespace(__name__=base.__name__, build_model=build_model, contador=contador)
