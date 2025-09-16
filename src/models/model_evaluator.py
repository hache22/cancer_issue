import numpy as np
from typing import Dict
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.utils.logger import logger

class ModelEvaluator:
    """
    Clase para evaluar los modelos de clustering utilizando un conjunto de métricas.
    """
    def evaluate_clusters(self, X_pca: np.ndarray, labels_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calcula las métricas de evaluación para cada modelo de clustering.

        Args:
            X_pca (np.ndarray): Datos reducidos por PCA sobre los que se generaron los clusters.
            labels_dict (Dict[str, np.ndarray]): Diccionario con las etiquetas de cada modelo.

        Returns:
            Dict[str, Dict[str, float]]: Un diccionario anidado con los puntajes para cada modelo.
        """
        scores = {}
        logger.info("Iniciando evaluación de modelos de clustering.")
        for model_name, labels in labels_dict.items():
            if len(np.unique(labels)) < 2:
                logger.warning(f"No se pueden calcular métricas para {model_name} porque tiene menos de 2 clusters.")
                continue

            scores[model_name] = {
                'Silhouette': silhouette_score(X_pca, labels),
                'Calinski-Harabasz': calinski_harabasz_score(X_pca, labels),
                'Davies-Bouldin': davies_bouldin_score(X_pca, labels)
            }
        
        logger.info("Evaluación de clusters completada.")
        return scores
