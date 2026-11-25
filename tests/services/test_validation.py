from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.services.validation import (
    InputValidationError,
    ModelOutputError,
    route_problems,
    validate_case,
    validate_cases_frame,
    validate_model_routes,
)

from .conftest import tramos_graph

G = tramos_graph()
NODES = set(G.nodes)
VEH = {f"VF-{i:02d}" for i in range(1, 16)}
END = date(2026, 9, 20)


def _case(**kw):
    base = {"caso_id": "C1", "vehiculo_id": "VF-01", "punto_inicial": "P1", "fecha_hora_inicio": "2026-09-25 08:00:00"}
    base.update(kw)
    return base


def test_caso_valido():
    c, err = validate_case(_case(), NODES, VEH, END)
    assert err is None and c.punto_inicial == "P1" and c.fecha_hora_inicio.hour == 8


@pytest.mark.parametrize(
    "campo,valor,texto",
    [
        ("caso_id", "", "caso_id vacío"),
        ("caso_id", "E 01", "caso_id inválido"),
        ("vehiculo_id", "ABC-123", "vehiculo_id inválido"),
        ("punto_inicial", "P99", "no existe"),
        ("punto_inicial", "", "punto_inicial vacío"),
        ("fecha_hora_inicio", "25/09/2026 08:00", "fecha_hora_inicio inválida"),
        ("fecha_hora_inicio", "2026-09-31 08:00:00", "fecha_hora_inicio inválida"),
    ],
)
def test_caso_invalido(campo, valor, texto):
    _, err = validate_case(_case(**{campo: valor}), NODES, VEH, END)
    assert err is not None and any(texto in m for m in err.mensajes)


def test_avisos_no_bloquean():
    c, err = validate_case(_case(vehiculo_id="VF-99", fecha_hora_inicio="2026-09-10 08:00:00"), NODES, VEH, END)
    assert err is None
    assert any("vehiculos.csv" in a for a in c.advertencias)
    assert any("entrenamiento" in a for a in c.advertencias)


def test_columnas_faltantes():
    df = pd.DataFrame([{"caso_id": "C1", "vehiculo_id": "VF-01", "punto_inicial": "P1"}])
    with pytest.raises(InputValidationError, match="fecha_hora_inicio"):
        validate_cases_frame(df, NODES, VEH, END)


def test_caso_id_duplicado():
    df = pd.DataFrame([_case(), _case(punto_inicial="P2")])
    validos, errores = validate_cases_frame(df, NODES, VEH, END)
    assert len(validos) == 1 and "duplicado" in errores[0].mensajes[-1] and errores[0].fila == 2


@pytest.mark.parametrize(
    "ruta,problema",
    [
        ("P1>P9", "tramo inexistente"),
        ("P1", "al menos 2"),
        ("P1>P10", "inexistentes"),
        ("P2>P5", "no inicia"),
        ("P1> P5", "espacios"),
        ("P1>>P5", "mal formada"),
    ],
)
def test_route_problems(ruta, problema):
    assert any(problema in p for p in route_problems(ruta, G, "P1"))


def test_ruta_valida():
    assert route_problems("P1>P5>P8>P9", G, "P1") == []


def test_filtra_duplicadas_invalidas_y_reordena():
    raw = [("P1>P2", 0.2), ("P1>P5", 0.5), ("P1>P5", 0.4), ("P1>P9", 0.9), ("P1>P4", 0.3)]
    rutas, avisos = validate_model_routes(raw, "P1", G, 3, 2)
    assert [r.ruta for r in rutas] == ["P1>P5", "P1>P4", "P1>P2"]
    assert [r.rango for r in rutas] == [1, 2, 3]
    assert any("duplicada" in a for a in avisos)
    assert any("tramo inexistente" in a for a in avisos)
    assert any("reordenaron" in a for a in avisos)


@pytest.mark.parametrize(
    "raw",
    [[("P1>P5", 1.0)], [("P1>P5", 1.0), ("P1>P5", 0.5)], [("P1>P5", float("nan")), ("P1>P2", 1)], "P1>P5", None],
)
def test_rutas_insuficientes(raw):
    with pytest.raises(ModelOutputError):
        validate_model_routes(raw, "P1", G, 3, 2)
