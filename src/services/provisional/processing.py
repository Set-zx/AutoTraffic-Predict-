"""PROVISIONAL — implementa el contrato ProcessingModule. Reemplazar por src.preprocessing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import networkx as nx
import pandas as pd


def load_network(data_dir: Path) -> nx.Graph:
    nodos = pd.read_csv(Path(data_dir) / "nodos.csv")
    tramos = pd.read_csv(Path(data_dir) / "tramos.csv")
    g = nx.Graph()
    for r in nodos.itertuples(index=False):
        g.add_node(r.punto_id, nombre=r.nombre, x_km=float(r.x_km), y_km=float(r.y_km))
    for r in tramos.itertuples(index=False):
        g.add_edge(
            r.origen,
            r.destino,
            tramo_id=r.tramo_id,
            tipo=r.tipo,
            longitud_km=float(r.longitud_km),
            vel_max_kmh=float(r.vel_max_kmh),
            tiempo_min=60.0 * float(r.longitud_km) / float(r.vel_max_kmh),
        )
    return g


def _reconstruct(seq: list[str], g: nx.Graph) -> tuple[list[str], bool]:
    """Rellena captaciones faltantes con el camino de menor tiempo entre puntos no vecinos."""
    out = [seq[0]]
    rec = False
    for a, b in zip(seq, seq[1:]):
        if a == b:
            continue
        if g.has_edge(a, b):
            out.append(b)
        else:
            out.extend(nx.shortest_path(g, a, b, weight="tiempo_min")[1:])
            rec = True
    return out, rec


def load_trips(data_dir: Path, start: date, end: date) -> pd.DataFrame:
    g = load_network(data_dir)
    cap = pd.read_csv(Path(data_dir) / "captaciones.csv", parse_dates=["fecha_hora"])
    cap = cap.sort_values(["recorrido_id", "fecha_hora", "captacion_id"])
    rows = []
    for rid, grp in cap.groupby("recorrido_id", sort=True):
        inicio = grp["fecha_hora"].iloc[0]
        if not (start <= inicio.date() <= end):
            continue
        ruta, rec = _reconstruct(list(grp["punto_id"]), g)
        rows.append(
            {
                "recorrido_id": rid,
                "vehiculo_id": grp["vehiculo_id"].iloc[0],
                "fecha_hora_inicio": inicio,
                "ruta": ">".join(ruta),
                "reconstruida": rec,
            }
        )
    return pd.DataFrame(rows)
