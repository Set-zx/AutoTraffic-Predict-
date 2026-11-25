"""Línea de comandos: genera predicciones.csv a partir de un CSV de casos.

    python -m src.services.cli                       # casos de ejemplo -> outputs/predicciones.csv
    python -m src.services.cli --casos casos.csv --salida outputs/predicciones.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_CASES, DEFAULT_OUTPUT, ServiceConfig
from .prediction_service import get_service
from .validation import AutoTrafficError, read_cases_csv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AutoTraffic: genera predicciones.csv (caso_id,rango,ruta)")
    p.add_argument("--casos", default=DEFAULT_CASES, type=Path,
                   help="CSV con caso_id,vehiculo_id,punto_inicial,fecha_hora_inicio (por defecto data/evaluation/casos_ejemplo.csv)")
    p.add_argument("--salida", default=DEFAULT_OUTPUT, type=Path, help="por defecto outputs/predicciones.csv")
    p.add_argument("--no-estricto", action="store_true", help="omite casos inválidos en lugar de detenerse (no genera archivo oficial)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    try:
        svc = get_service(ServiceConfig.from_env())
        casos = read_cases_csv(args.casos)
        lote = svc.predict_batch(casos, strict=not args.no_estricto)
        for e in lote.errores:
            print(f"ERROR {e}", file=sys.stderr)
        for r in lote.resultados:
            for a in r.advertencias:
                print(f"AVISO {r.caso_id}: {a}", file=sys.stderr)
        salida = svc.export(
            lote,
            args.salida,
            expected_case_ids=None if args.no_estricto else casos["caso_id"].str.strip(),
            allow_partial=args.no_estricto,
        )
    except AutoTrafficError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    i = svc.info
    print(
        f"{len(lote.resultados)} casos -> {salida} | backend={i.backend} modelo={i.model_source} "
        f"entrenamiento={i.train_start}..{i.train_end} ({i.n_trips} recorridos)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
