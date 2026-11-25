from __future__ import annotations

import pandas as pd
import pytest

from src.services import BatchResult, CaseError, CaseResult, ExportError, RankedRoute, export_predictions

from .conftest import tramos_graph

G = tramos_graph()
TS = pd.Timestamp("2026-09-25 08:00:00")


def _res(cid, inicio, *rutas):
    return CaseResult(cid, "VF-01", inicio, TS, tuple(RankedRoute(i, r, 1.0 / i) for i, r in enumerate(rutas, 1)))


def test_formato_exacto(tmp_path):
    lote = BatchResult((_res("E002", "P1", "P1>P5", "P1>P2", "P1>P4"), _res("E001", "P3", "P3>P2", "P3>P6")))
    out = export_predictions(lote, tmp_path / "predicciones.csv", graph=G, expected_case_ids=["E001", "E002"])
    texto = out.read_text(encoding="utf-8")
    assert texto.splitlines()[0] == "caso_id,rango,ruta"
    assert "\r" not in texto and "puntaje" not in texto
    df = pd.read_csv(out, dtype=str)
    assert list(df.columns) == ["caso_id", "rango", "ruta"]
    assert list(df["caso_id"]) == ["E002"] * 3 + ["E001"] * 2  # respeta el orden de entrada
    assert list(df["rango"]) == ["1", "2", "3", "1", "2"]


def test_rechaza_lote_con_errores(tmp_path):
    lote = BatchResult((_res("E001", "P3", "P3>P2", "P3>P6"),), (CaseError("E002", 2, ("x",)),))
    with pytest.raises(ExportError, match="con error"):
        export_predictions(lote, tmp_path / "p.csv")
    assert not (tmp_path / "p.csv").exists()


def test_rechaza_casos_faltantes(tmp_path):
    with pytest.raises(ExportError, match="faltan casos: E002"):
        export_predictions([_res("E001", "P3", "P3>P2", "P3>P6")], tmp_path / "p.csv", expected_case_ids=["E001", "E002"])


@pytest.mark.parametrize(
    "rutas,msg",
    [
        (("P3>P2", "P3>P2"), "duplicadas"),
        (("P3>P2",), "2 o 3"),
        (("P3>P2", "P3>P9"), "tramo inexistente"),
    ],
)
def test_rechaza_rutas_invalidas(tmp_path, rutas, msg):
    with pytest.raises(ExportError, match=msg):
        export_predictions([_res("E001", "P3", *rutas)], tmp_path / "p.csv", graph=G)


def test_rechaza_inicio_distinto(tmp_path):
    with pytest.raises(ExportError, match="no inicia"):
        export_predictions([_res("E001", "P1", "P3>P2", "P3>P6")], tmp_path / "p.csv")


def test_no_deja_archivo_a_medias(tmp_path):
    destino = tmp_path / "predicciones.csv"
    destino.write_text("previo\n")
    with pytest.raises(ExportError):
        export_predictions([_res("E001", "P3", "P3>P2")], destino, graph=G)
    assert destino.read_text() == "previo\n"
    assert list(tmp_path.iterdir()) == [destino]
