"""Pruebas de integración: procesamiento + modelo + validación + exportación.

Usan los módulos reales (src.preprocessing / src.prediction) si existen; si no, el backend
provisional. No verifican un valor de exactitud: solo el contrato y el formato.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.services import (
    DataContractError,
    InputValidationError,
    PredictionService,
    ServiceConfig,
    get_service,
)
from src.services.cli import main as cli_main
from src.services.validation import route_problems

from . import fakes
from .conftest import CASES_CSV, tramos_graph

G = tramos_graph()


@pytest.fixture
def servicio(svc_config) -> PredictionService:
    return PredictionService(svc_config)


def _assert_resultado_oficial(res, punto_inicial):
    assert 2 <= len(res.rutas) <= 3
    assert [r.rango for r in res.rutas] == list(range(1, len(res.rutas) + 1))
    rutas = [r.ruta for r in res.rutas]
    assert len(set(rutas)) == len(rutas)
    for r in rutas:
        assert r.split(">")[0] == punto_inicial
        assert route_problems(r, G, punto_inicial) == []


def test_no_requiere_soluciones(svc_config):
    assert not (svc_config.data_dir / "solucion_ejemplo.csv").exists()
    PredictionService(svc_config)  # no falla sin el archivo de soluciones


def test_entrena_solo_con_dias_1_a_20(servicio):
    assert servicio.info.train_start == date(2026, 9, 1)
    assert servicio.info.train_end == date(2026, 9, 20)
    assert servicio.info.n_trips == 166  # recorridos cuya primera captación es del 1 al 20


def test_config_rechaza_ventana_con_validacion():
    with pytest.raises(ValueError, match="2026-09-20"):
        ServiceConfig(train_end=date(2026, 9, 21))


def test_lote_completo_casos_ejemplo(servicio, casos_ejemplo, tmp_path):
    lote = servicio.predict_batch(casos_ejemplo)
    assert lote.ok and len(lote.resultados) == len(casos_ejemplo) == 41
    for res, (_, fila) in zip(lote.resultados, casos_ejemplo.iterrows()):
        assert res.caso_id == fila["caso_id"]
        _assert_resultado_oficial(res, fila["punto_inicial"])

    out = servicio.export(lote, tmp_path / "predicciones.csv", expected_case_ids=casos_ejemplo["caso_id"])
    df = pd.read_csv(out, dtype=str)
    assert list(df.columns) == ["caso_id", "rango", "ruta"]
    assert set(df["caso_id"]) == set(casos_ejemplo["caso_id"])
    assert not df.duplicated(["caso_id", "ruta"]).any()


def test_lote_desde_ruta_csv(servicio):
    lote = servicio.predict_batch(CASES_CSV)
    assert len(lote.resultados) == 41


def test_determinista(servicio, casos_ejemplo):
    a = servicio.predict_batch(casos_ejemplo).to_frame()
    b = servicio.predict_batch(casos_ejemplo).to_frame()
    pd.testing.assert_frame_equal(a, b)


def test_caso_individual_dict_y_series(servicio, caso_valido):
    r1 = servicio.predict_case(caso_valido)
    r2 = servicio.predict_case(pd.Series(caso_valido))
    _assert_resultado_oficial(r1, "P3")
    assert [r.ruta for r in r1.rutas] == [r.ruta for r in r2.rutas]


def test_caso_individual_invalido(servicio, caso_valido):
    with pytest.raises(InputValidationError, match="punto_inicial"):
        servicio.predict_case({**caso_valido, "punto_inicial": "P0"})


def test_lote_estricto_y_no_estricto(servicio, casos_ejemplo):
    malo = casos_ejemplo.copy()
    malo.loc[3, "fecha_hora_inicio"] = "ayer"
    malo.loc[5, "caso_id"] = malo.loc[4, "caso_id"]
    with pytest.raises(InputValidationError) as exc:
        servicio.predict_batch(malo)
    assert {e.fila for e in exc.value.errores} == {4, 6}

    lote = servicio.predict_batch(malo, strict=False)
    assert len(lote.resultados) == 39 and len(lote.errores) == 2


def test_rechaza_soluciones_como_entrada(servicio, casos_ejemplo):
    con_solucion = casos_ejemplo.assign(ruta_real="P1>P2")
    with pytest.raises(InputValidationError, match="soluciones"):
        servicio.predict_batch(con_solucion)


def test_no_existe_lectura_de_soluciones_en_servicios():
    from src.services.config import PROJECT_ROOT

    for f in (PROJECT_ROOT / "src" / "services").rglob("*.py"):
        assert "solucion_ejemplo" not in f.read_text(encoding="utf-8"), f


def test_columna_faltante(servicio, casos_ejemplo):
    with pytest.raises(InputValidationError, match="punto_inicial"):
        servicio.predict_batch(casos_ejemplo.drop(columns=["punto_inicial"]))


# ---------------------------------------------------------------- modelo ---
def test_entrena_una_sola_vez(svc_config, casos_ejemplo, caso_valido):
    mod = fakes.counting_model_module(svc_config)
    svc = PredictionService(svc_config, model_module=mod)
    svc.predict_batch(casos_ejemplo)
    svc.predict_batch(casos_ejemplo)
    svc.predict_case(caso_valido)
    assert mod.contador.builds == 1 and mod.contador.fits == 1


def test_get_service_reutiliza_instancia(svc_config):
    s1 = get_service(svc_config)
    s2 = get_service(svc_config)
    assert s1 is s2


def test_modelo_con_tramo_invalido_se_filtra(svc_config, caso_valido):
    mod = fakes.model_module([("P3>P9", 0.9), ("P3>P2>P5", 0.6), ("P3>P6", 0.4)])
    svc = PredictionService(svc_config, model_module=mod)
    res = svc.predict_case(caso_valido)
    assert [r.ruta for r in res.rutas] == ["P3>P2>P5", "P3>P6"]
    assert any("tramo inexistente" in a for a in res.advertencias)


def test_modelo_insuficiente_estricto_y_no_estricto(svc_config, casos_ejemplo):
    mod = fakes.model_module(lambda p: [(f"{p}>{p}", 1.0)])  # ruta inválida
    svc = PredictionService(svc_config, model_module=mod)
    with pytest.raises(InputValidationError, match="al menos 2"):
        svc.predict_batch(casos_ejemplo)
    lote = svc.predict_batch(casos_ejemplo, strict=False)
    assert not lote.resultados and len(lote.errores) == 41


def test_excepcion_del_modelo_se_aisla(svc_config, casos_ejemplo):
    mod = fakes.model_module(RuntimeError("boom"))
    svc = PredictionService(svc_config, model_module=mod)
    lote = svc.predict_batch(casos_ejemplo, strict=False)
    assert len(lote.errores) == 41 and "boom" in lote.errores[0].mensajes[0]


# ------------------------------------------------------- procesamiento ---
def test_detecta_fuga_de_datos(svc_config):
    def agrega_validacion(trips, _):
        extra = trips.iloc[[0]].copy()
        extra["recorrido_id"] = "FUGA"
        extra["fecha_hora_inicio"] = pd.Timestamp("2026-09-22 08:00:00")
        return pd.concat([trips, extra], ignore_index=True)

    proc = fakes.processing_con_trips(svc_config, agrega_validacion)
    with pytest.raises(DataContractError, match="fuga"):
        PredictionService(svc_config, processing=proc)


def test_detecta_saltos_sin_reconstruir(svc_config):
    def rompe(trips, _):
        t = trips.copy()
        t.loc[t.index[0], "ruta"] = "P1>P9"
        return t

    proc = fakes.processing_con_trips(svc_config, rompe)
    with pytest.raises(DataContractError, match="saltos"):
        PredictionService(svc_config, processing=proc)


# ------------------------------------------------------------------ CLI ---
def test_cli_genera_predicciones(svc_config, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AUTOTRAFFIC_DATA_DIR", str(svc_config.data_dir))
    monkeypatch.setenv("AUTOTRAFFIC_BACKEND", svc_config.backend)
    salida = tmp_path / "predicciones.csv"
    rc = cli_main(["--casos", str(CASES_CSV), "--salida", str(salida)])
    assert rc == 0
    assert salida.read_text(encoding="utf-8").splitlines()[0] == "caso_id,rango,ruta"
    assert "41 casos" in capsys.readouterr().out


def test_cli_falla_con_csv_invalido(svc_config, monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOTRAFFIC_DATA_DIR", str(svc_config.data_dir))
    monkeypatch.setenv("AUTOTRAFFIC_BACKEND", svc_config.backend)
    malo = tmp_path / "casos.csv"
    malo.write_text("caso_id,vehiculo_id,punto_inicial\nX1,VF-01,P1\n", encoding="utf-8")
    salida = tmp_path / "predicciones.csv"
    assert cli_main(["--casos", str(malo), "--salida", str(salida)]) == 1
    assert not salida.exists()
