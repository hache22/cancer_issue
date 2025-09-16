"""
Módulo de configuración para el proyecto de análisis de cáncer.

Centraliza las rutas de archivos, parámetros y listas de características
para facilitar la mantenibilidad y la configuración del pipeline.
"""

from pathlib import Path

# --- Rutas Principales ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Archivos de Datos ---
RAW_DATA_FILE = RAW_DATA_DIR / "cancer_issue.csv"

# --- Listas de Características ---
# Se definen aquí para no tenerlas hardcodeadas en el preprocesador.
# Estas serían las columnas originales del dataset.
# Nota: Las características generadas dinámicamente no se listan aquí.
CATEGORICAL_FEATURES = [
    "Gender",
    "Race/Ethnicity",
    "SmokingStatus",
    "Stage",
    "TreatmentType",
    "TreatmentResponse",
    "Recurrence",
    "GeneticMarker",
    "HospitalRegion",
    "FamilyHistory",
    "CancerType",
]

NUMERIC_FEATURES = [
    "Age",
    "BMI",
    "TumorSize",
    "SurvivalMonths",
]

# --- Parámetros del Modelo ---
CLUSTERING_PARAMS = {
    "n_clusters": 8,
    "random_state": 42,
}

# --- Parámetros de Preprocesamiento ---
OUTLIER_PARAMS = {
    "z_score_threshold": 3.0
}
