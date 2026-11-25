from collections import Counter, defaultdict
import networkx as nx
import pandas as pd
from .scoring import format_predictions


def _filter_train_period(train_trips: pd.DataFrame) -> pd.DataFrame:
    """Garantiza entrenamiento exclusivo del 1 al 20 de septiembre de 2026 (cero fugas)."""
    df = train_trips.copy()
    df["fecha_hora_inicio"] = pd.to_datetime(df["fecha_hora_inicio"])
    mask = (df["fecha_hora_inicio"] >= "2026-09-01") & (df["fecha_hora_inicio"] < "2026-09-21")
    return df.loc[mask].copy()


def fit(train_trips: pd.DataFrame, graph: nx.Graph) -> dict:
    """Entrena el modelo base de frecuencias históricas."""
    df = _filter_train_period(train_trips)

    by_vehicle_origin: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_origin: dict[str, Counter] = defaultdict(Counter)

    for _, row in df.iterrows():
        v_id = str(row["vehiculo_id"]).strip()
        origin = str(row["punto_inicial"]).strip()
        route = str(row["ruta"]).strip()
        by_vehicle_origin[(v_id, origin)][route] += 1
        by_origin[origin][route] += 1

    return {
        "model_name": "baseline_frecuencias",
        "by_vehicle_origin": dict(by_vehicle_origin),
        "by_origin": dict(by_origin),
        "n_train_trips": len(df),
    }


def predict(case: dict, model: dict, graph: nx.Graph, top_k: int = 3) -> dict:
    """Predice de 2 a 3 rutas ordenadas usando frecuencia por vehículo y respaldo general."""
    v_id = str(case["vehiculo_id"]).strip()
    origin = str(case["punto_inicial"]).strip()

    candidates: list[tuple[str, float, str]] = []

    # 1. Prioridad: historial del mismo vehículo desde el mismo origen
    veh_counts: Counter = model["by_vehicle_origin"].get((v_id, origin), Counter())
    total_veh = sum(veh_counts.values())
    if total_veh > 0:
        for route, count in veh_counts.most_common(top_k):
            candidates.append((route, (count / total_veh) * 1.0, "Frecuencia historica del vehiculo"))

    # 2. Respaldo: frecuencia general desde el mismo origen
    gen_counts: Counter = model["by_origin"].get(origin, Counter())
    total_gen = sum(gen_counts.values())
    if total_gen > 0:
        for route, count in gen_counts.most_common(top_k * 2):
            # Peso menor (0.35) para no desplazar rutas propias del vehículo
            candidates.append((route, (count / total_gen) * 0.35, "Respaldo por frecuencia general"))

    return format_predictions(case, candidates, graph, top_k=top_k)
