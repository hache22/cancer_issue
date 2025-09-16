
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils import config

# --- Fixtures para Pruebas ---

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Crea un DataFrame de ejemplo para las pruebas."""
    data = {
        'Age': [55, 62, 48, 70, 200], # 200 es un outlier
        'BMI': [23.5, 28.1, 21.0, 30.2, 50.0], # 50 es un outlier
        'TumorSize': [2.0, 3.5, 1.5, 4.0, 2.5],
        'SurvivalMonths': [24, 18, 36, 12, 30],
        'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
        'Race/Ethnicity': ['Caucasian', 'African American', 'Asian', 'Caucasian', 'Hispanic'],
        'SmokingStatus': ['Non-smoker', 'Smoker', 'Non-smoker', 'Former smoker', 'Non-smoker'],
        'Stage': ['II', 'III', 'I', 'IV', 'II'],
        'TreatmentType': ['Chemotherapy', 'Surgery', 'Radiation', 'Chemotherapy', 'Surgery'],
        'TreatmentResponse': ['Complete', 'Partial', 'Complete', 'No Response', 'Partial'],
        'Recurrence': ['No', 'Yes', 'No', 'Yes', 'No'],
        'GeneticMarker': ['Negative', 'Positive', 'Negative', 'Positive', 'Negative'],
        'HospitalRegion': ['North', 'South', 'West', 'East', 'North']
    }
    return pd.DataFrame(data)

# --- Pruebas para DataLoader ---

def test_load_data_success():
    """Prueba que DataLoader carga datos exitosamente desde un CSV de muestra."""
    sample_csv_path = Path(__file__).parent / "sample_data.csv"
    loader = DataLoader(file_path=sample_csv_path)
    df = loader.load_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 5

def test_load_data_file_not_found():
    """Prueba que DataLoader maneja correctamente un archivo no encontrado."""
    loader = DataLoader(file_path=Path("non_existent_file.csv"))
    df = loader.load_data()
    assert df is None

# --- Pruebas para DataPreprocessor ---

def test_preprocessor_fit_transform(sample_dataframe):
    """Prueba el flujo fit y transform del preprocesador."""
    preprocessor = DataPreprocessor()
    preprocessor.fit(sample_dataframe)
    X_processed, df_processed = preprocessor.transform(sample_dataframe)
    
    assert isinstance(X_processed, np.ndarray)
    assert isinstance(df_processed, pd.DataFrame)
    # Se espera que el outlier sea eliminado
    assert len(df_processed) == 4

def test_feature_creation(sample_dataframe):
    """Prueba la creación de nuevas características."""
    preprocessor = DataPreprocessor()
    df_featured = preprocessor._create_features(sample_dataframe)
    
    assert 'Age_BMI_Interaction' in df_featured.columns
    assert 'TumorSize_SurvivalRatio' in df_featured.columns
    assert 'BMI_Category' in df_featured.columns
    assert df_featured.loc[0, 'Age_BMI_Interaction'] == 55 * 23.5

def test_outlier_removal(sample_dataframe):
    """Prueba que la eliminación de outliers funciona."""
    preprocessor = DataPreprocessor()
    # El umbral por defecto es 3. El Z-score de 200 en Age será > 3.
    df_clean = preprocessor._handle_outliers(sample_dataframe)
    assert len(df_clean) == 4
    assert 200 not in df_clean['Age'].values
