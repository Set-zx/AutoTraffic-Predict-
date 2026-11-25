"""Servicio central de predicción de rutas.

Uso típico (Streamlit, CLI o pruebas)::

    from src.services.prediction_service import predict_case, predict_batch
    res = predict_case({"caso_id": "X1", "vehiculo_id": "VF-03",
                        "punto_inicial": "P1", "fecha_hora_inicio": "2026-09-25 07:30:00"})

El modelo se entrena (o se carga) una sola vez por proceso y se reutiliza.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import networkx as nx
import pandas as pd

from .config import ServiceConfig
from .contracts import (
    BatchResult,
    CaseError,
    CaseResult,
    ModelModule,
    ProcessingModule,
    RouteModel,
    ServiceInfo,
)
from .validation import (
    AutoTrafficError,
    InputValidationError,
    ModelOutputError,
    ValidatedCase,
    read_cases_csv,
    validate_case,
    validate_cases_frame,
    validate_graph,
    validate_model_routes,
    validate_trips_frame,
)

log = logging.getLogger("autotraffic.services")


def _import(module_name: str, rol: str, backend: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = (
            " Mientras se integra, puede usarse AUTOTRAFFIC_BACKEND=provisional."
            if backend == "real"
            else ""
        )
        raise AutoTrafficError(f"No se pudo importar el módulo de {rol} '{module_name}': {exc}.{hint}") from exc


class PredictionService:
    """Orquesta procesamiento (Daniel) + modelo (Omar) + validaciones."""

    def __init__(
        self,
        config: ServiceConfig | None = None,
        *,
        processing: ProcessingModule | ModuleType | None = None,
        model_module: ModelModule | ModuleType | None = None,
    ) -> None:
        t0 = time.perf_counter()
        self.config = config or ServiceConfig.from_env()
        cfg = self.config
        if not cfg.data_dir.is_dir():
            raise AutoTrafficError(f"No existe el directorio de datos {cfg.data_dir}")

        proc_name = cfg.resolved_processing_module if processing is None else getattr(processing, "__name__", type(processing).__name__)
        model_name = cfg.resolved_model_module if model_module is None else getattr(model_module, "__name__", type(model_module).__name__)
        self._processing = processing or _import(proc_name, "procesamiento", cfg.backend)
        self._model_module = model_module or _import(model_name, "modelo", cfg.backend)
        for fn in ("load_network", "load_trips"):
            if not callable(getattr(self._processing, fn, None)):
                raise AutoTrafficError(f"El módulo de procesamiento no expone {fn}()")
        if not callable(getattr(self._model_module, "build_model", None)):
            raise AutoTrafficError("El módulo del modelo no expone build_model()")
        if cfg.backend == "provisional":
            log.warning("Backend PROVISIONAL activo: no usar para la entrega final.")

        # 1. Red vial
        graph = self._processing.load_network(cfg.data_dir)
        validate_graph(graph)
        self.graph: nx.Graph = graph
        self.nodes: set[str] = set(graph.nodes)
        self.vehicles: set[str] = self._read_vehicle_ids(cfg.data_dir)

        # 2. Modelo: cargar si hay artefacto; si no, entrenar con días 1–20
        trips = validate_trips_frame(
            self._processing.load_trips(cfg.data_dir, cfg.train_start, cfg.train_end),
            graph,
            cfg.train_start,
            cfg.train_end,
        )
        self.model, source = self._load_or_train(trips)

        self.info = ServiceInfo(
            backend=cfg.backend,
            processing_module=proc_name,
            model_module=model_name,
            train_start=cfg.train_start,
            train_end=cfg.train_end,
            n_trips=len(trips),
            model_source=source,
            load_seconds=round(time.perf_counter() - t0, 3),
        )
        log.info("Servicio listo: %s", self.info)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_vehicle_ids(data_dir: Path) -> set[str]:
        path = data_dir / "vehiculos.csv"
        if not path.is_file():
            return set()
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        return set(df.get("vehiculo_id", pd.Series(dtype=str)).str.strip())

    def _load_or_train(self, trips: pd.DataFrame) -> tuple[RouteModel, str]:
        path = self.config.model_path
        loader = getattr(self._model_module, "load_model", None)
        if path is not None and path.is_file() and callable(loader):
            log.info("Cargando modelo desde %s", path)
            return loader(path, self.graph), "cargado"
        model = self._model_module.build_model(self.graph)
        for fn in ("fit", "predict"):
            if not callable(getattr(model, fn, None)):
                raise AutoTrafficError(f"build_model() devolvió un objeto sin {fn}()")
        model.fit(trips)
        saver = getattr(model, "save", None)
        if path is not None and callable(saver):
            path.parent.mkdir(parents=True, exist_ok=True)
            saver(path)
        return model, "entrenado"

    # ------------------------------------------------------------------ #
    def _predict_validated(self, case: ValidatedCase) -> CaseResult:
        cfg = self.config
        try:
            raw = self.model.predict(
                case.vehiculo_id, case.punto_inicial, case.fecha_hora_inicio, k=cfg.top_k
            )
        except Exception as exc:  # el modelo es código externo: aislar fallos por caso
            raise ModelOutputError(f"el modelo falló: {type(exc).__name__}: {exc}") from exc
        rutas, avisos = validate_model_routes(
            raw, case.punto_inicial, self.graph, cfg.top_k, cfg.min_routes
        )
        return CaseResult(
            caso_id=case.caso_id,
            vehiculo_id=case.vehiculo_id,
            punto_inicial=case.punto_inicial,
            fecha_hora_inicio=case.fecha_hora_inicio,
            rutas=rutas,
            advertencias=case.advertencias + avisos,
        )

    def predict_case(self, case: Mapping[str, Any] | pd.Series) -> CaseResult:
        """Predice un caso. Lanza InputValidationError o ModelOutputError."""
        raw = case.to_dict() if isinstance(case, pd.Series) else dict(case)
        validated, err = validate_case(raw, self.nodes, self.vehicles, self.config.train_end)
        if err:
            raise InputValidationError([err])
        assert validated is not None
        return self._predict_validated(validated)

    def predict_batch(self, cases: pd.DataFrame | str | Path, *, strict: bool = True) -> BatchResult:
        """Predice todos los casos de un DataFrame o CSV con formato oficial.

        strict=True: cualquier caso inválido o sin rutas suficientes detiene el lote
        (antes de predecir si el error es de entrada). strict=False: devuelve los
        casos correctos y la lista de errores (útil para la interfaz).
        """
        df = cases if isinstance(cases, pd.DataFrame) else read_cases_csv(cases)
        validos, errores = validate_cases_frame(df, self.nodes, self.vehicles, self.config.train_end)
        if strict and errores:
            raise InputValidationError(errores)

        resultados: list[CaseResult] = []
        for case in validos:
            try:
                resultados.append(self._predict_validated(case))
            except ModelOutputError as exc:
                err = CaseError(case.caso_id, None, (str(exc),))
                if strict:
                    raise InputValidationError([err]) from exc
                errores.append(err)
        return BatchResult(tuple(resultados), tuple(errores))

    def export(self, results: BatchResult | list[CaseResult], path: str | Path = "predicciones.csv", **kw: Any) -> Path:
        from .export_service import export_predictions

        return export_predictions(results, path, graph=self.graph, **kw)


# ---------------------------------------------------------------------- #
# Instancia única por proceso (Streamlit reutiliza el módulo entre recargas)
# ---------------------------------------------------------------------- #
_lock = threading.Lock()
_service: PredictionService | None = None


def get_service(config: ServiceConfig | None = None) -> PredictionService:
    """Devuelve la instancia compartida; la crea (entrena/carga) solo la primera vez."""
    global _service
    if _service is not None and (config is None or config == _service.config):
        return _service
    with _lock:
        if _service is None or (config is not None and config != _service.config):
            _service = PredictionService(config)
        return _service


def reset_service() -> None:
    """Descarta la instancia compartida (pruebas o cambio de configuración)."""
    global _service
    with _lock:
        _service = None


def predict_case(case: Mapping[str, Any] | pd.Series) -> CaseResult:
    return get_service().predict_case(case)


def predict_batch(cases_df: pd.DataFrame | str | Path, *, strict: bool = True) -> BatchResult:
    return get_service().predict_batch(cases_df, strict=strict)
