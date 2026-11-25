import networkx as nx


def parse_route(route_value: str | list[str]) -> list[str]:
    """Convierte 'P7>P4>P5' o una lista en una lista limpia de nodos."""
    if isinstance(route_value, list):
        return [str(n).strip() for n in route_value if str(n).strip()]
    return [n.strip() for n in str(route_value).split(">") if n.strip()]


def is_valid_route(route_nodes: list[str], start_node: str, graph: nx.Graph) -> bool:
    """Verifica que la ruta empiece en start_node, tenga al menos 2 nodos y use tramos válidos."""
    if len(route_nodes) < 2 or route_nodes[0] != start_node:
        return False
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        if not graph.has_edge(u, v):
            return False
    return True


def compute_route_distance_km(route_nodes: list[str], graph: nx.Graph) -> float:
    """Suma la longitud_km de los tramos del grafo de Villa Demo."""
    total = 0.0
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge = graph.edges[u, v]
        total += float(edge.get("longitud_km", 0.0))
    return round(total, 2)


def format_predictions(
    case: dict,
    scored_candidates: list[tuple[str, float, str]],
    graph: nx.Graph,
    top_k: int = 3,
) -> dict:
    """
    Construye el diccionario exacto del contrato técnico.
    scored_candidates: lista de tuplas (ruta_str, puntaje_bruto, criterio)
    """
    start_node = str(case["punto_inicial"]).strip()
    valid_unique = []
    seen_routes = set()

    for route_str, raw_score, criterio in scored_candidates:
        nodes = parse_route(route_str)
        canonical = ">".join(nodes)
        if canonical in seen_routes:
            continue
        if is_valid_route(nodes, start_node, graph):
            seen_routes.add(canonical)
            valid_unique.append((nodes, max(float(raw_score), 1e-6), criterio))
        if len(valid_unique) == top_k:
            break

    total_score = sum(item[1] for item in valid_unique)
    predicciones = []

    for idx, (nodes, raw_score, criterio) in enumerate(valid_unique, start=1):
        puntaje_rel = round(raw_score / total_score, 4) if total_score > 0 else 0.0
        predicciones.append({
            "rango": idx,
            "ruta": nodes,
            "puntaje": puntaje_rel,
            "distancia_km": compute_route_distance_km(nodes, graph),
            "criterio": criterio,
        })

    # Ajuste fino para que los puntajes relativos sumen 1.00
    if predicciones:
        diff = round(1.0 - sum(p["puntaje"] for p in predicciones), 4)
        predicciones[0]["puntaje"] = round(predicciones[0]["puntaje"] + diff, 4)

    result = {
        "caso_id": str(case.get("caso_id", "")),
        "vehiculo_id": str(case["vehiculo_id"]),
        "punto_inicial": start_node,
        "predicciones": predicciones,
    }
    if len(predicciones) < 2:
        result["advertencia"] = "Evidencia histórica insuficiente para generar 2 o 3 rutas distintas."

    return result
