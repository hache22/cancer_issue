
import numpy as np
from typing import Dict, Tuple, Any

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from src.utils import config
from src.utils.logger import logger

class ModelTrainer:
    """
    Clase para entrenar los modelos de clustering y aplicar PCA.
    """
    def __init__(
        self, 
        clustering_params: Dict[str, Any] = config.CLUSTERING_PARAMS,
        pca_n_components: float = 0.95
    ):
        """
        Inicializa el ModelTrainer.

        Args:
            clustering_params (Dict[str, Any]): Parámetros para los modelos de clustering.
            pca_n_components (float): Varianza a retener por PCA.
        """
        self.params = clustering_params
        self.pca_n_components = pca_n_components
        self.pca: PCA = None
        self.models: Dict[str, Any] = {}

    def apply_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica PCA para reducción de dimensionalidad.
        """
        logger.info(f"Aplicando PCA para retener {self.pca_n_components * 100}% de la varianza...")
        self.pca = PCA(n_components=self.pca_n_components)
        X_pca = self.pca.fit_transform(X)
        logger.info(f"PCA aplicado. Dimensiones reducidas a: {X_pca.shape[1]}")
        return X_pca

    def train_clustering_models(self, X_pca: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Entrena un conjunto predefinido de modelos de clustering.
        """
        n_clusters = self.params['n_clusters']
        random_state = self.params['random_state']

        defined_models = {
            "KMeans": KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
            "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
            "GMM": GaussianMixture(n_components=n_clusters, random_state=random_state)
        }

        labels: Dict[str, np.ndarray] = {}

        for name, model in defined_models.items():
            logger.info(f"Entrenando modelo {name}...")
            labels[name] = model.fit_predict(X_pca)
            self.models[name] = model
        
        logger.info("Todos los modelos de clustering han sido entrenados.")
        return self.models, labels
