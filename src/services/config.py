"""Configuración de los servicios. Se puede sobrescribir con variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Estructura acordada del repositorio
RAW_DIR = PROJECT_ROOT / "data" / "raw"                # CSV oficiales (Noe)
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"  # casos/solución de ejemplo: SOLO evaluación
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CASES = EVALUATION_DIR / "casos_ejemplo.csv"
DEFAULT_OUTPUT = OUTPUTS_DIR / "predicciones.csv"

# Módulos de los compañeros (contrato en docs/contrato_servicios.md)
DEFAULT_PROCESSING = "src.preprocessing"  # Daniel
DEFAULT_MODEL = "src.prediction"          # Omar

# Reglas del reto: entrenar solo con 1–20 sep; 21–24 es validación; 25–30 reservado.
TRAIN_START = date(2026, 9, 1)
TRAIN_END = date(2026, 9, 20)

BACKENDS = ("real", "provisional")
PROVISIONAL_PROCESSING = "src.services.provisional.processing"
PROVISIONAL_MODEL = "src.services.provisional.model"


@dataclass(frozen=True)
class ServiceConfig:
    data_dir: Path = RAW_DIR
    backend: str = "real"
    processing_module: str = DEFAULT_PROCESSING
    model_module: str = DEFAULT_MODEL
    train_start: date = TRAIN_START
    train_end: date = TRAIN_END
    top_k: int = 3
    min_routes: int = 2
    model_path: Path | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend debe ser uno de {BACKENDS}, no {self.backend!r}")
        if self.train_start < TRAIN_START or self.train_end > TRAIN_END:
            raise ValueError(
                "El entrenamiento solo puede usar del 2026-09-01 al 2026-09-20 "
                "(21–24 es validación; 25–30 está reservado para los jueces)."
            )
        if self.train_start > self.train_end:
            raise ValueError("train_start no puede ser posterior a train_end")
        if not 2 <= self.min_routes <= self.top_k <= 3:
            raise ValueError("Se requiere 2 <= min_routes <= top_k <= 3")

    @property
    def resolved_processing_module(self) -> str:
        return PROVISIONAL_PROCESSING if self.backend == "provisional" else self.processing_module

    @property
    def resolved_model_module(self) -> str:
        return PROVISIONAL_MODEL if self.backend == "provisional" else self.model_module

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Variables: AUTOTRAFFIC_DATA_DIR, AUTOTRAFFIC_BACKEND, AUTOTRAFFIC_PROCESSING_MODULE,
        AUTOTRAFFIC_MODEL_MODULE, AUTOTRAFFIC_MODEL_PATH."""
        env = os.environ
        kwargs: dict = {}
        if v := env.get("AUTOTRAFFIC_DATA_DIR"):
            kwargs["data_dir"] = Path(v)
        if v := env.get("AUTOTRAFFIC_BACKEND"):
            kwargs["backend"] = v.strip().lower()
        if v := env.get("AUTOTRAFFIC_PROCESSING_MODULE"):
            kwargs["processing_module"] = v
        if v := env.get("AUTOTRAFFIC_MODEL_MODULE"):
            kwargs["model_module"] = v
        if v := env.get("AUTOTRAFFIC_MODEL_PATH"):
            kwargs["model_path"] = Path(v)
        return cls(**kwargs)
