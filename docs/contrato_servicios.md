# Contrato técnico · Servicios centrales (`src/services/`)

**Responsable:** Set (líder técnico / backend) · **Consumidores:** interfaz Streamlit, CLI, pruebas
**Proveedores:** Daniel (procesamiento, `src/preprocessing/`) · Omar (modelo, `src/prediction/`)

La fuente de verdad del contrato en código es `src/services/contracts.py`. Si alguien necesita cambiarlo, se acuerda en equipo y se actualizan juntos este documento y ese archivo.

---

## 1. Reglas del reto que aplican los servicios

| Regla | Dónde se aplica |
|---|---|
| Entrenar solo con los días 1 al 20 de septiembre de 2026 | `ServiceConfig` rechaza otra ventana; `validate_trips_frame` rechaza recorridos fuera de ella (protección contra fuga de datos) |
| Los días 21 al 24 son de validación y no se entrenan con ellos | Mismo control |
| `solucion_ejemplo.csv` no entra al modelo | Vive en `data/evaluation/`, que los servicios nunca leen para entrenar; una prueba ejecuta todo sin ese archivo, otra verifica que ningún servicio lo menciona y la entrada se rechaza si trae la columna `ruta_real` |
| 2 o 3 rutas por caso, distintas, que inicien en `punto_inicial` y usen tramos válidos | `validate_model_routes` y `export_predictions` |
| Salida con exactamente `caso_id,rango,ruta` | `export_predictions` |
| Los puntajes son relativos, no probabilidades | El campo se llama `puntaje` / `puntaje_relativo` y no se exporta |

---

## 2. Contrato con Daniel (procesamiento)

Módulo importable `src.preprocessing`. Su `src/preprocessing/__init__.py` debe exponer (puede reexportarlas desde `loader.py`, `graph_builder.py` y `reconstruction.py`):

```python
def load_network(data_dir: Path) -> networkx.Graph
def load_trips(data_dir: Path, start: date, end: date) -> pandas.DataFrame
```

**`load_network`**
- Devuelve un `nx.Graph` **no dirigido**, porque los tramos son bidireccionales.
- Nodos `P1` a `P9`.
- Atributos de arista recomendados: `tramo_id`, `tipo`, `longitud_km`, `vel_max_kmh`. Los servicios solo usan la adyacencia; el modelo puede usar los atributos.

**`load_trips`**: un renglón por recorrido cuya **primera captación** cae entre `start` y `end`, ambos inclusive.

| Columna | Tipo | Nota |
|---|---|---|
| `recorrido_id` | str | Único |
| `vehiculo_id` | str | `VF-NN` |
| `fecha_hora_inicio` | datetime64 | Fecha y hora de la primera captación |
| `ruta` | str | `P1>P5>P8`, completa, con los **saltos por captación faltante ya reconstruidos** |

Se permiten columnas extra, por ejemplo `reconstruida: bool`.

**Qué valida el servicio sobre estos datos.** Lanza `DataContractError` si:
- el grafo es dirigido o está vacío,
- faltan columnas,
- hay `recorrido_id` duplicado,
- hay fechas inválidas o fuera de la ventana de entrenamiento,
- alguna ruta usa un tramo inexistente (es decir, un salto sin reconstruir).

## 3. Contrato con Omar (modelo)

Módulo importable `src.prediction`. Su `src/prediction/__init__.py` debe exponer (puede reexportarlas desde `model.py`, `baseline.py` y `scoring.py`):

```python
def build_model(graph: nx.Graph) -> RouteModel

class RouteModel:
    def fit(self, trips: pd.DataFrame) -> Any            # recibe la salida de load_trips
    def predict(self, vehiculo_id: str, punto_inicial: str,
                fecha_hora_inicio: pd.Timestamp, k: int = 3
               ) -> list[tuple[str, float]]               # [(ruta, puntaje), ...]
```

- `predict` devuelve hasta `k` pares `(ruta, puntaje)`, ordenados de mayor a menor puntaje.
- El puntaje es un número finito y **relativo**. Si Omar lo calibra como probabilidad, debe documentar el método y la evidencia de validación en su módulo.
- El modelo debe intentar entregar siempre **3 rutas distintas**, porque la métrica es Top-3. Si conoce menos rutas históricas, el respaldo con la red vial es responsabilidad del modelo. El servicio **no inventa rutas**.
- **Persistencia (opcional).** Si existen `load_model(path, graph)` en el módulo y `model.save(path)` en el modelo, y se define `AUTOTRAFFIC_MODEL_PATH`:
  - si el archivo existe, se carga;
  - si no existe, se entrena y se guarda.

**Qué valida el servicio sobre la salida del modelo:**
- Descarta, con un aviso, las rutas que no inician en el punto solicitado, que tienen nodos o tramos inexistentes, que tienen menos de 2 puntos, que están duplicadas o cuyo puntaje no es numérico o no es finito.
- Reordena por puntaje si llegaron desordenadas.
- Si quedan menos de 2 rutas válidas, lanza `ModelOutputError`.
- Si `predict` lanza una excepción, el fallo se aísla en ese caso.

## 4. API de los servicios (lo que consume Streamlit)

```python
from src.services import get_service, predict_case, predict_batch, export_predictions

svc = get_service()          # entrena o carga UNA vez por proceso y reutiliza la instancia
res = predict_case({"caso_id": "X1", "vehiculo_id": "VF-03",
                    "punto_inicial": "P1", "fecha_hora_inicio": "2026-09-25 07:30:00"})
lote = predict_batch(df_o_ruta_csv, strict=True)
export_predictions(lote, "outputs/predicciones.csv", graph=svc.graph,
                   expected_case_ids=df["caso_id"])   # o: svc.export(lote, ...)
```

| Función | Devuelve | Errores |
|---|---|---|
| `predict_case(case)` (`dict` o `pd.Series`) | `CaseResult(caso_id, vehiculo_id, punto_inicial, fecha_hora_inicio, rutas, advertencias)` | `InputValidationError`, `ModelOutputError` |
| `predict_batch(cases_df \| path, strict=True)` | `BatchResult(resultados, errores)`; `.to_frame()` para mostrar una tabla con `puntaje_relativo` | Con `strict=True`, `InputValidationError` con **todas** las filas inválidas. Con `strict=False`, los errores se reportan en `.errores` |
| `export_predictions(results, path="predicciones.csv", graph=None, expected_case_ids=None, allow_partial=False)` | `Path` | `ExportError` si el lote tiene errores, faltan casos, hay rutas repetidas, los rangos no son consecutivos o hay tramos inválidos |

`CaseResult.rutas` es una tupla de `RankedRoute(rango, ruta, puntaje)`.

**Validación de la entrada:**
- Columnas obligatorias: `caso_id`, `vehiculo_id`, `punto_inicial`, `fecha_hora_inicio`. Las columnas extra se ignoran, salvo `ruta_real`, que provoca un error porque indicaría que se pasaron soluciones como entrada.
- `caso_id`: hasta 32 caracteres entre letras, dígitos, `_` y `-`, sin repetirse.
- `vehiculo_id`: formato `VF-NN`. Si no aparece en `vehiculos.csv` solo se genera un aviso.
- `punto_inicial`: debe existir en la red.
- `fecha_hora_inicio`: formato `AAAA-MM-DD HH:MM:SS` y fecha real. Si cae dentro de la ventana de entrenamiento se genera un aviso, porque esa evaluación no sería honesta.
- Los errores indican la fila del CSV (1 = primera fila de datos).

**El archivo `predicciones.csv`:**
- Codificación UTF-8, saltos de línea LF y sin índice.
- Los casos aparecen en el mismo orden que en la entrada.
- La escritura es atómica: si la validación falla, no se crea ni se sobrescribe ningún archivo.

**En Streamlit:**

```python
import streamlit as st
from src.services import get_service

@st.cache_resource
def servicio():
    return get_service()
```

## 5. Configuración (variables de entorno)

| Variable | Valor por defecto | Uso |
|---|---|---|
| `AUTOTRAFFIC_DATA_DIR` | `data/raw` | Carpeta con `nodos.csv`, `tramos.csv`, `vehiculos.csv` y `captaciones.csv` |
| `AUTOTRAFFIC_BACKEND` | `real` | `provisional` solo mientras se integra |
| `AUTOTRAFFIC_PROCESSING_MODULE` | `src.preprocessing` | Si Daniel usa otro nombre de módulo |
| `AUTOTRAFFIC_MODEL_MODULE` | `src.prediction` | Si Omar usa otro nombre de módulo |
| `AUTOTRAFFIC_MODEL_PATH` | (vacío) | Artefacto del modelo para cargar o guardar |

## 6. Backend provisional (temporal)

`src/services/provisional/` implementa los dos contratos para poder trabajar antes de la integración:
- La reconstrucción de saltos usa el camino de menor tiempo entre los dos puntos.
- El modelo es una línea base por frecuencia histórica, ponderada por la hora, el tipo de día (entre semana o fin de semana) y el vehículo, con respaldo en la red vial.

**No simula resultados**: calcula con los CSV de los días 1 al 20. Solo se activa con `AUTOTRAFFIC_BACKEND=provisional`, que nunca es el valor por defecto.

Referencia medida para comparar el modelo de Omar (entrenando con los días 1 al 20 y evaluando contra `data/evaluation/solucion_ejemplo.csv`, 41 casos de los días 21 al 24): **Top-3 33/41 (80.5 %)** y **Top-1 30/41 (73.2 %)**. Es una medición sobre la muestra de validación, no una estimación del resultado con los casos reservados de los jueces. Los puntajes del modelo provisional están normalizados a suma 1 por caso, pero **son relativos, no probabilidades calibradas**.

**Se elimina antes de la entrega**, y `python -m pytest -m entrega` falla mientras exista.

---

## 7. Pruebas

```bash
python -m pytest                 # unitarias e integración (las de entrega se omiten)
python -m pytest tests/services  # solo las de servicios
python -m pytest -m entrega      # verificación previa a la entrega
```

- Las pruebas de servicios viven en `tests/services/` y no dependen de `pyproject.toml` ni de otros archivos compartidos: el marcador `entrega` se registra en `tests/services/conftest.py`.
- Las de integración usan automáticamente `src.preprocessing` y `src.prediction` cuando ambos existen; si falta alguno, usan el backend provisional.
- `tests/services/fakes.py` contiene dobles de prueba que provocan salidas inválidas a propósito. Solo los usan las pruebas y no generan predicciones entregables.
- `pytest -m entrega` comprueba: módulos reales presentes, carpeta `provisional/` eliminada, servicio real arrancando y los 41 casos de ejemplo exportados en formato oficial.

---

## 8. Ubuntu: instalación y ejecución

Requisito del reto: Python 3.12. Comprueba la versión con `python3 --version`. Si la de tu sistema no es la 3.12, instala esa versión aparte (por ejemplo con `uv python install 3.12` o el PPA *deadsnakes*) y crea el entorno con ella.

```bash
sudo apt update && sudo apt install -y git python3-venv python3-pip
git clone https://github.com/Set-zx/AutoTraffic-Predict-.git
cd AutoTraffic-Predict-
python3.12 -m venv .venv          # o python3 -m venv .venv si ya es 3.12
source .venv/bin/activate
pip install -r requirements.txt   # archivo de Noe

python -m pytest                                  # pruebas
python -m src.services.cli                        # ejemplo -> outputs/predicciones.csv
python -m src.services.cli --casos ruta/a/casos.csv --salida outputs/predicciones.csv
AUTOTRAFFIC_BACKEND=provisional python -m src.services.cli   # mientras no estén Daniel y Omar
```

Dependencias que usan los servicios (para el `requirements.txt` de Noe): `pandas`, `numpy`, `networkx` y `pytest`. Los servicios no usan Plotly, Streamlit ni scikit-learn.

---

## 9. Integración mediante Git

Cada responsable trabaja en su rama y solo toca su carpeta. Los cambios entran a `main` por *pull request*.

```bash
# Set (servicios)
git checkout main && git pull
git checkout -b feature/services
# copiar src/services/, tests/services/ y docs/contrato_servicios.md
git add src/services tests/services docs/contrato_servicios.md
git commit -m "services: predict_case, predict_batch y export_predictions con validación"
git push -u origin feature/services   # abrir PR hacia main en GitHub
```

Orden sugerido de integración:
1. **Servicios** (esta rama). Con `AUTOTRAFFIC_BACKEND=provisional`, Itzel ya puede conectar Streamlit.
2. **Daniel** (`feature/preprocessing`): `src/preprocessing/__init__.py` con `load_network` y `load_trips`. Antes del PR: `python -m pytest tests/services`.
3. **Omar** (`feature/prediction`): `src/prediction/__init__.py` con `build_model`. Antes del PR: `python -m pytest tests/services`. Comparar su Top-3 con la referencia de la sección 6.
4. Con ambos en `main`: rama `chore/quitar-provisional`, `git rm -r src/services/provisional`, `python -m pytest -m entrega` en verde y PR.

Si alguien cambia una firma del contrato, se acuerda en equipo y en el mismo PR se actualizan este documento y `src/services/contracts.py`.
