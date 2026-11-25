"""Exportación de predicciones al formato oficial: caso_id,rango,ruta."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from .contracts import OUTPUT_COLUMNS, BatchResult, CaseResult
from .validation import ExportError, parse_route, validate_output_frame


def _as_results(results: BatchResult | Iterable[CaseResult], allow_partial: bool) -> list[CaseResult]:
    if isinstance(results, BatchResult):
        if results.errores and not allow_partial:
            raise ExportError(
                f"el lote tiene {len(results.errores)} caso(s) con error; el archivo oficial "
                "debe cubrir todos los casos (use allow_partial=True solo para pruebas)"
            )
        return list(results.resultados)
    if isinstance(results, CaseResult):
        return [results]
    lista = list(results)
    if not all(isinstance(r, CaseResult) for r in lista):
        raise ExportError("se esperaban objetos CaseResult")
    return lista


def results_to_frame(results: BatchResult | Iterable[CaseResult], *, allow_partial: bool = False) -> pd.DataFrame:
    """DataFrame con exactamente las columnas oficiales, en el orden de entrada de los casos."""
    lista = _as_results(results, allow_partial)
    ids = [r.caso_id for r in lista]
    if len(ids) != len(set(ids)):
        raise ExportError("hay caso_id repetidos entre los resultados")
    rows = []
    for res in lista:
        for ruta in res.rutas:
            if parse_route(ruta.ruta)[0] != res.punto_inicial:
                raise ExportError(f"{res.caso_id}: la ruta {ruta.ruta} no inicia en {res.punto_inicial}")
            rows.append((res.caso_id, int(ruta.rango), ruta.ruta))
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def export_predictions(
    results: BatchResult | Iterable[CaseResult],
    path: str | Path = "predicciones.csv",
    *,
    graph: nx.Graph | None = None,
    expected_case_ids: Iterable[str] | None = None,
    allow_partial: bool = False,
) -> Path:
    """Valida y escribe ``predicciones.csv`` (UTF-8, sin índice, saltos LF).

    - ``graph``: si se pasa, re-verifica que cada tramo exista.
    - ``expected_case_ids``: si se pasa, exige que estén todos y solo esos casos.
    La escritura es atómica: nunca queda un archivo a medias.
    """
    df = results_to_frame(results, allow_partial=allow_partial)
    validate_output_frame(df, graph=graph, expected_case_ids=expected_case_ids)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".predicciones-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False, lineterminator="\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
