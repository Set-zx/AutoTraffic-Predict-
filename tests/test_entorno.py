from pathlib import Path
import pandas as pd

def test_archivos_datos_existen():
    for r in ["data/raw/captaciones.csv", "data/raw/nodos.csv", "data/raw/tramos.csv", "data/raw/vehiculos.csv", "data/evaluation/casos_ejemplo.csv", "data/evaluation/solucion_ejemplo.csv"]:
        assert Path(r).exists()

def test_conteos_oficiales():
    assert len(pd.read_csv("data/raw/captaciones.csv")) == 737
    assert len(pd.read_csv("data/raw/nodos.csv")) == 9
    assert len(pd.read_csv("data/raw/tramos.csv")) == 16
    assert len(pd.read_csv("data/raw/vehiculos.csv")) == 15
    assert len(pd.read_csv("data/evaluation/casos_ejemplo.csv")) == 41
