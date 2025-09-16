
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.model_explainer import ModelExplainer
from src.utils import config

# --- Fixtures para Pruebas ---

@pytest.fixture
def sample_processed_data() -> np.ndarray:
    """Crea un array de numpy de ejemplo, simulando datos preprocesados."""
    # 50 muestras, 10 características
    return np.random.rand(50, 10)

@pytest.fixture
def sample_clustered_dataframe() -> pd.DataFrame:
    """Crea un DataFrame con datos y una columna de cluster."""
    data = {
        'Age': np.random.randint(40, 80, 50),
        'BMI': np.random.uniform(20, 35, 50),
        'TumorSize': np.random.uniform(1, 5, 50),
        'SurvivalMonths': np.random.randint(10, 40, 50),
        'Cluster_KMeans': np.random.randint(0, 3, 50)
    }
    return pd.DataFrame(data)

# --- Pruebas para ModelTrainer ---

def test_trainer_apply_pca(sample_processed_data):
    """Prueba que PCA se aplica y reduce las dimensiones."""
    trainer = ModelTrainer()
    X_pca = trainer.apply_pca(sample_processed_data)
    assert X_pca.shape[0] == sample_processed_data.shape[0]
    assert X_pca.shape[1] < sample_processed_data.shape[1]

def test_trainer_train_models(sample_processed_data):
    """Prueba que los modelos de clustering se entrenan y devuelven resultados."""
    trainer = ModelTrainer()
    X_pca = trainer.apply_pca(sample_processed_data)
    models, labels = trainer.train_clustering_models(X_pca)
    
    assert 'KMeans' in models
    assert 'Agglomerative' in models
    assert 'GMM' in models
    assert 'KMeans' in labels
    assert len(labels['KMeans']) == sample_processed_data.shape[0]

# --- Pruebas para ModelEvaluator ---

def test_evaluator_calculates_scores(sample_processed_data):
    """Prueba que el evaluador calcula un diccionario de métricas."""
    # Generar etiquetas de prueba
    labels_dict = {
        'KMeans': np.random.randint(0, 3, sample_processed_data.shape[0]),
        'GMM': np.random.randint(0, 3, sample_processed_data.shape[0])
    }
    evaluator = ModelEvaluator()
    scores = evaluator.evaluate_clusters(sample_processed_data, labels_dict)
    
    assert 'KMeans' in scores
    assert 'Silhouette' in scores['KMeans']
    assert 'Calinski-Harabasz' in scores['KMeans']

# --- Pruebas para ModelExplainer ---

def test_explainer_runs_without_error(mocker, sample_clustered_dataframe, sample_processed_data):
    """Prueba que los métodos de ModelExplainer se ejecutan sin errores."""
    # Mock para evitar que se guarden los gráficos
    mocker.patch('src.models.model_explainer.ModelExplainer.save_figure')

    output_dir = Path("tests/temp_figures")
    explainer = ModelExplainer(output_dir=output_dir, best_model_name='KMeans')

    # Ejecutar todos los métodos para asegurar que no hay errores
    explainer.create_cluster_profiles(sample_clustered_dataframe)
    explainer.plot_pca_clusters(sample_processed_data, sample_clustered_dataframe['Cluster_KMeans'])
    explainer.plot_feature_distribution(sample_clustered_dataframe, 'Age')
    explainer.plot_categorical_distribution(sample_clustered_dataframe, 'Cluster_KMeans') # Usando una col existente
