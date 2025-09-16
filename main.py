
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.model_explainer import ModelExplainer
from src.utils import config
from src.utils.logger import logger

def main():
    """
    Función principal para ejecutar el pipeline de análisis de clustering de cáncer.
    """
    logger.info("--- Iniciando pipeline de análisis de clustering de cáncer ---")

    # 1. Carga de Datos
    logger.info("--- 1. Iniciando Carga de Datos ---")
    data_loader = DataLoader(file_path=config.RAW_DATA_FILE)
    df = data_loader.load_data()
    if df is None:
        logger.error("Finalizando pipeline debido a un error en la carga de datos.")
        return
    logger.info("Carga de datos completada.\n")

    # 2. Preprocesamiento de Datos
    logger.info("--- 2. Iniciando Preprocesamiento de Datos ---")
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)
    X_processed, df_processed = preprocessor.transform(df)
    logger.info("Preprocesamiento de datos completado.\n")

    # 3. Entrenamiento de Modelos
    logger.info("--- 3. Iniciando Entrenamiento de Modelos ---")
    trainer = ModelTrainer(clustering_params=config.CLUSTERING_PARAMS)
    X_pca = trainer.apply_pca(X_processed)
    models, labels = trainer.train_clustering_models(X_pca)
    logger.info("Entrenamiento de modelos completado.\n")

    # 4. Evaluación de Modelos
    logger.info("--- 4. Iniciando Evaluación de Modelos ---")
    evaluator = ModelEvaluator()
    scores = evaluator.evaluate_clusters(X_pca, labels)
    
    for model_name, metrics in scores.items():
        logger.info(f"--- Resultados de {model_name} ---")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
    logger.info("Evaluación de modelos completada.\n")

    # Seleccionar el mejor modelo
    best_model_name = 'KMeans'
    logger.info(f"Mejor modelo seleccionado: {best_model_name}\n")

    # 5. Interpretación y Visualización
    logger.info(f"--- 5. Iniciando Interpretación para {best_model_name} ---")
    explainer = ModelExplainer(output_dir=config.FIGURES_DIR, best_model_name=best_model_name)
    
    df_processed[f'Cluster_{best_model_name}'] = labels[best_model_name]

    explainer.create_cluster_profiles(df_processed)
    explainer.plot_pca_clusters(X_pca, labels[best_model_name])
    explainer.plot_feature_distribution(df_processed, 'SurvivalMonths')
    explainer.plot_categorical_distribution(df_processed, 'CancerType')
    explainer.plot_categorical_distribution(df_processed, 'TreatmentResponse')
    logger.info("Interpretación y visualización completadas.\n")

    logger.info("--- Pipeline completado exitosamente ---")

if __name__ == '__main__':
    main()
