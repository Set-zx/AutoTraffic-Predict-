"""Validaciones de entrada (casos), de los contratos internos y de la salida."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import pandas as pd

from .contracts import (
    CASE_COLUMNS,
    DATETIME_FORMAT,
    OUTPUT_COLUMNS,
    ROUTE_SEP,
    TRIP_COLUMNS,
    CaseError,
    RankedRoute,
)

CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
VEHICLE_ID_RE = re.compile(r"^VF-\d{2}$")
POINT_ID_RE = re.compile(r"^P\d+$")


# --------------------------------------------------------------------------- #
# Excepciones
# --------------------------------------------------------------------------- #
class AutoTrafficError(Exception):
    """Base de los errores de los servicios."""


class InputValidationError(AutoTrafficError):
    def __init__(self, errores: Sequence[CaseError]):
        self.errores = tuple(errores)
        detalle = "\n".join(f"  - {e}" for e in self.errores[:20])
        extra = f"\n  ... y {len(self.errores) - 20} más" if len(self.errores) > 20 else ""
        super().__init__(f"{len(self.errores)} caso(s) inválido(s):\n{detalle}{extra}")


class DataContractError(AutoTrafficError):
    """El módulo de procesamiento devolvió datos que no cumplen el contrato."""


class ModelOutputError(AutoTrafficError):
    """El modelo devolvió rutas insuficientes o inválidas para un caso."""


class ExportError(AutoTrafficError):
    """El resultado no puede exportarse con el formato oficial."""


# --------------------------------------------------------------------------- #
# Casos de entrada
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ValidatedCase:
    caso_id: str
    vehiculo_id: str
    punto_inicial: str
    fecha_hora_inicio: pd.Timestamp
    advertencias: tuple[str, ...] = ()


def read_cases_csv(path: str | Path) -> pd.DataFrame:
    """Lee un CSV de casos como texto (sin conversiones implícitas)."""
    path = Path(path)
    if not path.is_file():
        raise InputValidationError([CaseError(None, None, (f"no existe el archivo {path}",))])
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InputValidationError([CaseError(None, None, (f"CSV ilegible: {exc}",))]) from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


# Columnas que delatan que se pasó un archivo de soluciones como entrada.
SOLUTION_COLUMNS: tuple[str, ...] = ("ruta_real",)


def check_case_columns(df: pd.DataFrame) -> None:
    if fuga := [c for c in SOLUTION_COLUMNS if c in df.columns]:
        raise InputValidationError(
            [CaseError(None, None, (f"la entrada contiene {', '.join(fuga)}: las soluciones no pueden ser entrada del modelo",))]
        )
    faltantes = [c for c in CASE_COLUMNS if c not in df.columns]
    if faltantes:
        raise InputValidationError(
            [CaseError(None, None, (f"faltan columnas obligatorias: {', '.join(faltantes)}",))]
        )
    if df.columns.duplicated().any():
        raise InputValidationError([CaseError(None, None, ("hay columnas duplicadas",))])


def _parse_datetime(value: Any) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, str):
        try:
            ts = pd.Timestamp(datetime.strptime(value.strip(), DATETIME_FORMAT))
        except ValueError:
            return None
    else:
        return None
    if pd.isna(ts) or ts.tzinfo is not None:
        return None
    return ts


def validate_case(
    raw: Mapping[str, Any],
    nodes: set[str],
    known_vehicles: set[str],
    train_end: date,
    fila: int | None = None,
) -> tuple[ValidatedCase | None, CaseError | None]:
    errores: list[str] = []
    avisos: list[str] = []

    def text(col: str) -> str:
        v = raw.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return str(v).strip()

    faltantes = [c for c in CASE_COLUMNS if c not in raw]
    if faltantes:
        return None, CaseError(None, fila, (f"faltan campos: {', '.join(faltantes)}",))

    caso_id = text("caso_id")
    vehiculo_id = text("vehiculo_id")
    punto = text("punto_inicial")

    if not caso_id:
        errores.append("caso_id vacío")
    elif not CASE_ID_RE.match(caso_id):
        errores.append(f"caso_id inválido {caso_id!r} (solo letras, dígitos, '_' o '-', máx. 32)")

    if not vehiculo_id:
        errores.append("vehiculo_id vacío")
    elif not VEHICLE_ID_RE.match(vehiculo_id):
        errores.append(f"vehiculo_id inválido {vehiculo_id!r} (formato esperado VF-NN)")
    elif known_vehicles and vehiculo_id not in known_vehicles:
        avisos.append(f"vehiculo_id {vehiculo_id} no aparece en vehiculos.csv")

    if not punto:
        errores.append("punto_inicial vacío")
    elif not POINT_ID_RE.match(punto) or punto not in nodes:
        errores.append(f"punto_inicial {punto!r} no existe en la red")

    ts = _parse_datetime(raw.get("fecha_hora_inicio"))
    if ts is None:
        errores.append(
            f"fecha_hora_inicio inválida {raw.get('fecha_hora_inicio')!r} "
            f"(formato esperado AAAA-MM-DD HH:MM:SS)"
        )
    elif ts.date() <= train_end:
        avisos.append("fecha dentro del periodo de entrenamiento: no sirve como evaluación honesta")

    if errores:
        return None, CaseError(caso_id or None, fila, tuple(errores))
    assert ts is not None
    return ValidatedCase(caso_id, vehiculo_id, punto, ts, tuple(avisos)), None


def validate_cases_frame(
    df: pd.DataFrame, nodes: set[str], known_vehicles: set[str], train_end: date
) -> tuple[list[ValidatedCase], list[CaseError]]:
    check_case_columns(df)
    if df.empty:
        raise InputValidationError([CaseError(None, None, ("el archivo no contiene casos",))])

    validos: list[ValidatedCase] = []
    errores: list[CaseError] = []
    vistos: dict[str, int] = {}
    for i, raw in enumerate(df[list(CASE_COLUMNS)].to_dict(orient="records"), start=1):
        caso, err = validate_case(raw, nodes, known_vehicles, train_end, fila=i)
        cid = str(raw.get("caso_id", "")).strip()
        if cid and cid in vistos:
            msg = f"caso_id {cid} duplicado (primera aparición en fila {vistos[cid]})"
            err = CaseError(cid, i, (err.mensajes if err else ()) + (msg,))
            caso = None
        elif cid:
            vistos[cid] = i
        if err:
            errores.append(err)
        elif caso:
            validos.append(caso)
    return validos, errores


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
def parse_route(ruta: str) -> list[str]:
    return [p.strip() for p in str(ruta).split(ROUTE_SEP)]


def route_problems(ruta: Any, graph: nx.Graph, punto_inicial: str | None = None) -> list[str]:
    """Lista de problemas de una ruta (vacía si es válida)."""
    if not isinstance(ruta, str) or not ruta.strip():
        return ["ruta vacía o no es texto"]
    if ruta != ruta.strip() or " " in ruta:
        return [f"ruta {ruta!r} contiene espacios"]
    puntos = parse_route(ruta)
    problemas: list[str] = []
    if any(p == "" for p in puntos):
        return [f"ruta {ruta!r} mal formada"]
    if len(puntos) < 2:
        problemas.append(f"ruta {ruta!r} debe tener al menos 2 puntos")
    desconocidos = [p for p in puntos if p not in graph]
    if desconocidos:
        problemas.append(f"ruta {ruta!r} con puntos inexistentes: {', '.join(desconocidos)}")
        return problemas
    if punto_inicial is not None and puntos[0] != punto_inicial:
        problemas.append(f"ruta {ruta!r} no inicia en {punto_inicial}")
    for a, b in zip(puntos, puntos[1:]):
        if not graph.has_edge(a, b):
            problemas.append(f"ruta {ruta!r} usa el tramo inexistente {a}-{b}")
    return problemas


def validate_model_routes(
    raw_routes: Any,
    punto_inicial: str,
    graph: nx.Graph,
    top_k: int,
    min_routes: int,
) -> tuple[tuple[RankedRoute, ...], tuple[str, ...]]:
    """Filtra y ordena la salida del modelo. Lanza ModelOutputError si quedan < min_routes."""
    if raw_routes is None or isinstance(raw_routes, (str, bytes)):
        raise ModelOutputError("el modelo debe devolver una lista de pares (ruta, puntaje)")
    avisos: list[str] = []
    candidatas: list[tuple[str, float]] = []
    vistas: set[str] = set()
    for item in list(raw_routes):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            avisos.append(f"elemento descartado (no es par ruta, puntaje): {item!r}")
            continue
        ruta, puntaje = item
        try:
            puntaje = float(puntaje)
        except (TypeError, ValueError):
            avisos.append(f"ruta {ruta!r} descartada: puntaje no numérico")
            continue
        if not math.isfinite(puntaje):
            avisos.append(f"ruta {ruta!r} descartada: puntaje no finito")
            continue
        probs = route_problems(ruta, graph, punto_inicial)
        if probs:
            avisos.extend(f"descartada: {p}" for p in probs)
            continue
        if ruta in vistas:
            avisos.append(f"ruta duplicada descartada: {ruta}")
            continue
        vistas.add(ruta)
        candidatas.append((ruta, puntaje))

    ordenadas = sorted(candidatas, key=lambda rp: -rp[1])  # estable: respeta empates del modelo
    if [r for r, _ in ordenadas] != [r for r, _ in candidatas]:
        avisos.append("el modelo no devolvió las rutas ordenadas por puntaje; se reordenaron")
    ordenadas = ordenadas[:top_k]
    if len(ordenadas) < min_routes:
        raise ModelOutputError(
            f"el modelo produjo {len(ordenadas)} ruta(s) válida(s) distinta(s); se requieren "
            f"al menos {min_routes}. Detalle: {'; '.join(avisos) or 'sin rutas'}"
        )
    rutas = tuple(RankedRoute(i, r, p) for i, (r, p) in enumerate(ordenadas, start=1))
    return rutas, tuple(avisos)


# --------------------------------------------------------------------------- #
# Contrato del módulo de procesamiento
# --------------------------------------------------------------------------- #
def validate_graph(graph: Any) -> None:
    if not isinstance(graph, nx.Graph):
        raise DataContractError("load_network debe devolver un networkx.Graph")
    if graph.is_directed():
        raise DataContractError("load_network debe devolver un grafo NO dirigido (tramos bidireccionales)")
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise DataContractError("la red está vacía")
    malos = [n for n in graph.nodes if not isinstance(n, str) or not POINT_ID_RE.match(n)]
    if malos:
        raise DataContractError(f"identificadores de nodo inválidos: {malos[:5]}")


def validate_trips_frame(
    trips: Any, graph: nx.Graph, train_start: date, train_end: date
) -> pd.DataFrame:
    if not isinstance(trips, pd.DataFrame):
        raise DataContractError("load_trips debe devolver un pandas.DataFrame")
    faltantes = [c for c in TRIP_COLUMNS if c not in trips.columns]
    if faltantes:
        raise DataContractError(f"load_trips: faltan columnas {faltantes}")
    if trips.empty:
        raise DataContractError("load_trips devolvió 0 recorridos")
    if trips["recorrido_id"].duplicated().any():
        raise DataContractError("load_trips: recorrido_id duplicado")
    fechas = pd.to_datetime(trips["fecha_hora_inicio"], errors="coerce")
    if fechas.isna().any():
        raise DataContractError("load_trips: fecha_hora_inicio con valores inválidos")
    dias = fechas.dt.date
    fuera = trips.loc[(dias < train_start) | (dias > train_end), "recorrido_id"]
    if not fuera.empty:
        raise DataContractError(
            f"load_trips devolvió {len(fuera)} recorrido(s) fuera de {train_start}..{train_end} "
            f"(fuga de datos de validación/reservados). Ej.: {list(fuera[:3])}"
        )
    invalidas = []
    for rid, ruta in zip(trips["recorrido_id"], trips["ruta"]):
        probs = route_problems(ruta, graph)
        if probs:
            invalidas.append(f"{rid}: {probs[0]}")
    if invalidas:
        raise DataContractError(
            "load_trips devolvió rutas con tramos inválidos (¿saltos sin reconstruir?): "
            + "; ".join(invalidas[:5])
        )
    out = trips.copy()
    out["fecha_hora_inicio"] = fechas
    return out


# --------------------------------------------------------------------------- #
# Salida oficial
# --------------------------------------------------------------------------- #
def validate_output_frame(
    df: pd.DataFrame,
    graph: nx.Graph | None = None,
    expected_case_ids: Iterable[str] | None = None,
) -> None:
    if list(df.columns) != list(OUTPUT_COLUMNS):
        raise ExportError(f"columnas deben ser exactamente {','.join(OUTPUT_COLUMNS)}")
    if df.empty:
        raise ExportError("no hay predicciones para exportar")
    problemas: list[str] = []
    for cid, grupo in df.groupby("caso_id", sort=False):
        rangos = list(grupo["rango"])
        if not 2 <= len(grupo) <= 3:
            problemas.append(f"{cid}: {len(grupo)} rutas (se requieren 2 o 3)")
        if rangos != list(range(1, len(grupo) + 1)):
            problemas.append(f"{cid}: rangos {rangos} no son 1..n consecutivos")
        if grupo["ruta"].duplicated().any():
            problemas.append(f"{cid}: rutas duplicadas")
        iniciales = {parse_route(r)[0] for r in grupo["ruta"]}
        if len(iniciales) != 1:
            problemas.append(f"{cid}: las rutas no comparten el mismo punto inicial")
        if graph is not None:
            for r in grupo["ruta"]:
                problemas.extend(f"{cid}: {p}" for p in route_problems(r, graph))
    if expected_case_ids is not None:
        esperados = set(expected_case_ids)
        presentes = set(df["caso_id"])
        if faltan := sorted(esperados - presentes):
            problemas.append(f"faltan casos: {', '.join(faltan[:10])}")
        if sobran := sorted(presentes - esperados):
            problemas.append(f"casos no solicitados: {', '.join(sobran[:10])}")
    if problemas:
        raise ExportError("salida inválida:\n  - " + "\n  - ".join(problemas[:20]))
