"""Contratos técnicos entre los servicios y los módulos de Daniel (procesamiento) y Omar (modelo).

Este archivo es la fuente de verdad de las interfaces. Cualquier cambio se acuerda
en equipo y se documenta en ``docs/contrato_servicios.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import networkx as nx
import pandas as pd

# --------------------------------------------------------------------------- #
# Formatos oficiales
# --------------------------------------------------------------------------- #
CASE_COLUMNS: tuple[str, ...] = ("caso_id", "vehiculo_id", "punto_inicial", "fecha_hora_inicio")
OUTPUT_COLUMNS: tuple[str, ...] = ("caso_id", "rango", "ruta")
TRIP_COLUMNS: tuple[str, ...] = ("recorrido_id", "vehiculo_id", "fecha_hora_inicio", "ruta")
ROUTE_SEP = ">"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- #
# Contrato con el módulo de procesamiento (Daniel)
# --------------------------------------------------------------------------- #
@runtime_checkable
class ProcessingModule(Protocol):
    """Módulo importable (por defecto ``src.preprocessing``, Daniel) con estas dos funciones."""

    def load_network(self, data_dir: Path) -> nx.Graph:
        """Grafo NO dirigido con los 9 nodos y los 16 tramos de ``tramos.csv``.

        Nodos: id ``P1``..``P9``. Atributos de arista recomendados:
        ``tramo_id``, ``tipo``, ``longitud_km``, ``vel_max_kmh``.
        """
        ...

    def load_trips(self, data_dir: Path, start: date, end: date) -> pd.DataFrame:
        """Un renglón por recorrido cuya primera captación cae en [start, end] (inclusive).

        Columnas obligatorias (ver ``TRIP_COLUMNS``):
        - ``recorrido_id`` (str)
        - ``vehiculo_id`` (str)
        - ``fecha_hora_inicio`` (datetime64): fecha de la primera captación
        - ``ruta`` (str, ``P1>P5>P8``): secuencia COMPLETA, con los saltos por
          captaciones faltantes ya reconstruidos sobre tramos válidos.
        Columnas adicionales están permitidas (por ejemplo ``reconstruida``).
        """
        ...


# --------------------------------------------------------------------------- #
# Contrato con el módulo del modelo (Omar)
# --------------------------------------------------------------------------- #
@runtime_checkable
class RouteModel(Protocol):
    def fit(self, trips: pd.DataFrame) -> Any:
        """Entrena con el DataFrame de ``load_trips`` (solo días 1 a 20)."""
        ...

    def predict(
        self,
        vehiculo_id: str,
        punto_inicial: str,
        fecha_hora_inicio: pd.Timestamp,
        k: int = 3,
    ) -> Sequence[tuple[str, float]]:
        """Hasta ``k`` pares ``(ruta, puntaje)`` ordenados de mayor a menor puntaje.

        ``ruta`` usa el formato ``P1>P5>P8``. ``puntaje`` es un número finito y
        relativo: solo ordena rutas; no se interpreta como probabilidad calibrada.
        """
        ...


@runtime_checkable
class ModelModule(Protocol):
    """Módulo importable (por defecto ``src.prediction``, Omar).

    Obligatorio: ``build_model(graph) -> RouteModel``.
    Opcional (persistencia): ``load_model(path, graph) -> RouteModel`` y
    ``RouteModel.save(path)``.
    """

    def build_model(self, graph: nx.Graph) -> RouteModel: ...


# --------------------------------------------------------------------------- #
# Resultados que exponen los servicios (los consume Streamlit)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RankedRoute:
    rango: int
    ruta: str
    puntaje: float  # puntaje relativo del modelo, NO probabilidad calibrada

    @property
    def puntos(self) -> list[str]:
        return self.ruta.split(ROUTE_SEP)


@dataclass(frozen=True, slots=True)
class CaseResult:
    caso_id: str
    vehiculo_id: str
    punto_inicial: str
    fecha_hora_inicio: pd.Timestamp
    rutas: tuple[RankedRoute, ...]
    advertencias: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseError:
    caso_id: str | None
    fila: int | None  # número de fila de datos en el CSV (1 = primera fila tras el encabezado)
    mensajes: tuple[str, ...]

    def __str__(self) -> str:
        donde = f"fila {self.fila}" if self.fila is not None else "caso"
        ident = f" ({self.caso_id})" if self.caso_id else ""
        return f"{donde}{ident}: " + "; ".join(self.mensajes)


@dataclass(frozen=True)
class BatchResult:
    resultados: tuple[CaseResult, ...]
    errores: tuple[CaseError, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errores

    def to_frame(self, with_scores: bool = True) -> pd.DataFrame:
        """Tabla larga para la interfaz. Para el archivo oficial usar ``export_service``."""
        rows = []
        for res in self.resultados:
            for r in res.rutas:
                row = {"caso_id": res.caso_id, "rango": r.rango, "ruta": r.ruta}
                if with_scores:
                    row["puntaje_relativo"] = r.puntaje
                rows.append(row)
        cols = list(OUTPUT_COLUMNS) + (["puntaje_relativo"] if with_scores else [])
        return pd.DataFrame(rows, columns=cols)


@dataclass(frozen=True, slots=True)
class ServiceInfo:
    backend: str
    processing_module: str
    model_module: str
    train_start: date
    train_end: date
    n_trips: int
    model_source: str  # "entrenado" | "cargado"
    load_seconds: float
