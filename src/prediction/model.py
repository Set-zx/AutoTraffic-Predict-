from collections import Counter, defaultdict
import networkx as nx
import pandas as pd
from .baseline import _filter_train_period
from .scoring import format_predictions


def _get_time_slot(ts: pd.Timestamp) -> str:
    """Agrupa la hora en franjas operativas."""
    hour = ts.hour
    if 6 <= hour < 12:
        return "manana"
    if 12 <= hour < 18:
        return "tarde"
    if 18 <= hour < 23:
        return "noche"
    return "madrugada"


def _get_day_type(ts: pd.Timestamp) -> str:
    """Distingue día laboral (0-4) de fin de semana (5-6)."""
    return "fin_semana" if ts.dayofweek >= 5 else "entre_semana"


def fit(train_trips: pd.DataFrame, graph: nx.Graph) -> dict:
    """Entrena el modelo jerárquico con vehículo, origen, franja horaria y tipo de día."""
    df = _filter_train_period(train_trips)

    counts = {
        "veh_orig_slot_day": defaultdict(Counter),  # Nivel 1 (más específico)
        "veh_orig_slot": defaultdict(Counter),      # Nivel 2
        "veh_orig_day": defaultdict(Counter),       # Nivel 3
        "veh_orig": defaultdict(Counter),           # Nivel 4
        "orig_slot": defaultdict(Counter),          # Nivel 5
        "orig": defaultdict(Counter),               # Nivel 6 (respaldo global)
    }

    for _, row in df.iterrows():
        v_id = str(row["vehiculo_id"]).strip()
        origin = str(row["punto_inicial"]).strip()
        route = str(row["ruta"]).strip()
        ts = pd.to_datetime(row["fecha_hora_inicio"])
        slot = _get_time_slot(ts)
        day_type = _get_day_type(ts)

        counts["veh_orig_slot_day"][(v_id, origin, slot, day_type)][route] += 1
        counts["veh_orig_slot"][(v_id, origin, slot)][route] += 1
        counts["veh_orig_day"][(v_id, origin, day_type)][route] += 1
        counts["veh_orig"][(v_id, origin)][route] += 1
        counts["orig_slot"][(origin, slot)][route] += 1
        counts["orig"][origin][route] += 1

    return {
        "model_name": "patron_espacio_temporal_suavizado",
        "counts": {k: dict(v) for k, v in counts.items()},
        "n_train_trips": len(df),
    }


def predict(case: dict, model: dict, graph: nx.Graph, top_k: int = 3) -> dict:
    """Combina los 6 niveles con suavizado de Laplace y pesos jerárquicos."""
    v_id = str(case["vehiculo_id"]).strip()
    origin = str(case["punto_inicial"]).strip()
    ts = pd.to_datetime(case["fecha_hora_inicio"])
    slot = _get_time_slot(ts)
    day_type = _get_day_type(ts)

    c = model["counts"]
    levels = [
        (c["veh_orig_slot_day"].get((v_id, origin, slot, day_type), Counter()), 0.40, "Patron exacto (vehiculo + horario + dia)"),
        (c["veh_orig_slot"].get((v_id, origin, slot), Counter()), 0.25, "Habito del vehiculo en franja horaria"),
        (c["veh_orig_day"].get((v_id, origin, day_type), Counter()), 0.15, "Habito del vehiculo por tipo de dia"),
        (c["veh_orig"].get((v_id, origin), Counter()), 0.12, "Historial general del vehiculo"),
        (c["orig_slot"].get((origin, slot), Counter()), 0.05, "Tendencia general por franja horaria"),
        (c["orig"].get(origin, Counter()), 0.03, "Frecuencia general desde el origen"),
    ]

    # Reunir todas las rutas candidatas vistas desde ese origen
    all_routes = set()
    for counter, _, _ in levels:
        all_routes.update(counter.keys())

    alpha = 0.5  # Parámetro de suavizado estadístico para muestras pequeñas
    k_routes = max(len(all_routes), 1)

    scored = []
    for route in all_routes:
        total_score = 0.0
        best_criterio = "Frecuencia general desde el origen"
        best_contrib = -1.0

        for counter, weight, label in levels:
            n_total = sum(counter.values())
            if n_total == 0:
                continue
            prob_smoothed = (counter.get(route, 0) + alpha) / (n_total + alpha * k_routes)
            contrib = weight * prob_smoothed
            total_score += contrib
            if counter.get(route, 0) > 0 and contrib > best_contrib:
                best_contrib = contrib
                best_criterio = label

        scored.append((route, total_score, best_criterio))

    scored.sort(key=lambda x: x[1], reverse=True)
    return format_predictions(case, scored, graph, top_k=top_k)
