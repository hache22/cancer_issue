
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from src.utils.logger import logger

class ModelExplainer:
    """
    Clase para interpretar y visualizar los resultados del clustering.
    """
    def __init__(self, output_dir: Path, best_model_name: str = 'KMeans'):
        """
        Inicializa el ModelExplainer.

        Args:
            output_dir (Path): Directorio donde se guardarán las visualizaciones.
            best_model_name (str): Nombre del modelo seleccionado para la explicación.
        """
        self.output_dir = output_dir
        self.best_model_name = best_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_cluster_profiles(self, df_clustered: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Crea perfiles para cada cluster basados en las características promedio.
        """
        cluster_col = f'Cluster_{self.best_model_name}'
        if cluster_col not in df_clustered.columns:
            logger.error(f"La columna de cluster '{cluster_col}' no se encuentra en el DataFrame.")
            return None

        agg_dict = {
            'Age': 'mean',
            'BMI': 'mean',
            'TumorSize': 'mean',
            'SurvivalMonths': 'mean',
            'TreatmentResponse': lambda x: x.mode()[0] if not x.mode().empty else 'N/A',
            'Stage': lambda x: x.mode()[0] if not x.mode().empty else 'N/A',
            'CancerType': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
        }
        
        valid_agg_dict = {k: v for k, v in agg_dict.items() if k in df_clustered.columns}

        logger.info("Creando perfiles de clusters...")
        cluster_profiles = df_clustered.groupby(cluster_col).agg(valid_agg_dict).round(2)
        
        logger.info("--- Perfiles de Clusters ---")
        logger.info(f"\n{cluster_profiles.to_string()}")
        return cluster_profiles

    def save_figure(self, fig: plt.Figure, filename: str):
        """
        Guarda una figura de matplotlib en el directorio de salida.
        """
        path = self.output_dir / filename
        try:
            fig.savefig(path)
            plt.close(fig)
            logger.info(f"Gráfico guardado en: {path}")
        except Exception as e:
            logger.error(f"No se pudo guardar el gráfico {filename}: {e}")

    def plot_pca_clusters(self, X_pca: np.ndarray, labels: np.ndarray):
        """
        Crea y guarda una visualización de los clusters en un espacio 2D de PCA.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7, ax=ax)
        ax.set_title(f'Visualización de Clusters ({self.best_model_name}) usando PCA')
        ax.set_xlabel('Primer Componente Principal')
        ax.set_ylabel('Segundo Componente Principal')
        ax.legend(title='Cluster')
        self.save_figure(fig, "pca_clusters.png")

    def plot_feature_distribution(self, df_clustered: pd.DataFrame, feature: str):
        """
        Crea y guarda la distribución de una característica numérica por cluster.
        """
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=df_clustered, x=f'Cluster_{self.best_model_name}', y=feature, ax=ax)
        ax.set_title(f'Distribución de {feature} por Cluster')
        self.save_figure(fig, f"{feature.lower()}_distribution.png")

    def plot_categorical_distribution(self, df_clustered: pd.DataFrame, feature: str):
        """
        Crea y guarda la distribución de una característica categórica por cluster.
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        crosstab = pd.crosstab(df_clustered[f'Cluster_{self.best_model_name}'], df_clustered[feature], normalize='index')
        crosstab.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Distribución de {feature} por Cluster (%)')
        ax.set_ylabel('Porcentaje')
        ax.set_xlabel('Cluster')
        ax.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self.save_figure(fig, f"{feature.lower()}_categorical_distribution.png")
