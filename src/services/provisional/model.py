"""PROVISIONAL — línea base que implementa el contrato ModelModule. Reemplazar por src.prediction.

Puntaje de una ruta histórica r para (vehículo v, punto p, fecha t):
    suma sobre recorridos de entrenamiento que inician en p y siguieron r de
    K_hora(Δh) · w_dia · w_vehiculo
con K_hora gaussiano (σ = 1.5 h, diferencia horaria circular), w_dia = 1 si coincide
el tipo de día (entre semana / fin de semana) y 0.3 si no, y w_vehiculo = 1 para el
mismo vehículo y 0.15 para el resto de la flota. Si faltan rutas, se completan con
caminos de menor tiempo desde p. Los puntajes se normalizan a suma 1 pero son
RELATIVOS, no probabilidades calibradas.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


class FrequencyRouteModel:
    def __init__(self, graph: nx.Graph, sigma_horas: float = 1.5, peso_otro_dia: float = 0.3, peso_flota: float = 0.15):
        self.graph = graph
        self.sigma = sigma_horas
        self.peso_otro_dia = peso_otro_dia
        self.peso_flota = peso_flota
        self._trips: pd.DataFrame | None = None

    def fit(self, trips: pd.DataFrame) -> "FrequencyRouteModel":
        t = pd.to_datetime(trips["fecha_hora_inicio"])
        self._trips = pd.DataFrame(
            {
                "vehiculo_id": trips["vehiculo_id"].to_numpy(),
                "punto_inicial": trips["ruta"].str.split(">").str[0].to_numpy(),
                "hora": (t.dt.hour + t.dt.minute / 60.0).to_numpy(),
                "finde": (t.dt.dayofweek >= 5).to_numpy(),
                "ruta": trips["ruta"].to_numpy(),
            }
        )
        return self

    def _fallback(self, p: str) -> list[str]:
        dist, paths = nx.single_source_dijkstra(self.graph, p, weight="tiempo_min")
        destinos = sorted((d, n) for n, d in dist.items() if n != p)
        return [">".join(paths[n]) for _, n in destinos]

    def predict(self, vehiculo_id: str, punto_inicial: str, fecha_hora_inicio: pd.Timestamp, k: int = 3):
        if self._trips is None:
            raise RuntimeError("modelo no entrenado")
        df = self._trips[self._trips["punto_inicial"] == punto_inicial]
        h = fecha_hora_inicio.hour + fecha_hora_inicio.minute / 60.0
        finde = fecha_hora_inicio.dayofweek >= 5
        scores: dict[str, float] = defaultdict(float)
        if not df.empty:
            dh = np.abs(df["hora"].to_numpy() - h)
            dh = np.minimum(dh, 24.0 - dh)
            w = np.exp(-0.5 * (dh / self.sigma) ** 2)
            w *= np.where(df["finde"].to_numpy() == finde, 1.0, self.peso_otro_dia)
            w *= np.where(df["vehiculo_id"].to_numpy() == vehiculo_id, 1.0, self.peso_flota)
            for ruta, wi in zip(df["ruta"], w):
                scores[ruta] += float(wi)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        if len(ranked) < k:
            minimo = min((s for _, s in ranked), default=1.0)
            usados = {r for r, _ in ranked}
            extra = [r for r in self._fallback(punto_inicial) if r not in usados]
            for i, r in enumerate(extra[: k - len(ranked)]):
                ranked.append((r, minimo * 1e-3 / (i + 1)))
        total = sum(s for _, s in ranked) or 1.0
        return [(r, s / total) for r, s in ranked]


def build_model(graph: nx.Graph) -> FrequencyRouteModel:
    return FrequencyRouteModel(graph)
