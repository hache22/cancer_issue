
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import zscore
from typing import List, Tuple, Optional

from src.utils import config
from src.utils.logger import logger

class DataPreprocessor:
    """
    Clase para preprocesar los datos, incluyendo ingeniería de características,
    manejo de nulos, outliers y transformaciones.
    """
    def __init__(
        self, 
        numeric_features: List[str] = config.NUMERIC_FEATURES,
        categorical_features: List[str] = config.CATEGORICAL_FEATURES,
        outlier_threshold: float = config.OUTLIER_PARAMS["z_score_threshold"]
    ):
        """
        Inicializa el DataPreprocessor.

        Args:
            numeric_features (List[str]): Lista de columnas numéricas.
            categorical_features (List[str]): Lista de columnas categóricas.
            outlier_threshold (float): Umbral de Z-score para detectar outliers.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.outlier_threshold = outlier_threshold
        self._preprocessor: Optional[ColumnTransformer] = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea nuevas características a partir de las existentes.
        """
        df_copy = df.copy()
        if 'Age' in df_copy.columns and 'BMI' in df_copy.columns:
            df_copy['Age_BMI_Interaction'] = df_copy['Age'] * df_copy['BMI']
        if 'TumorSize' in df_copy.columns and 'SurvivalMonths' in df_copy.columns:
            df_copy['TumorSize_SurvivalRatio'] = df_copy['TumorSize'] / (df_copy['SurvivalMonths'] + 1e-6)
        if 'BMI' in df_copy.columns:
            df_copy['BMI_Category'] = pd.cut(df_copy['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            df_copy['BMI_Category'] = df_copy['BMI_Category'].cat.add_categories(['Unknown']).fillna('Unknown')
        return df_copy

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina outliers de las características numéricas usando Z-score.
        """
        df_copy = df.copy()
        existing_numeric = [feat for feat in self.numeric_features if feat in df_copy.columns]
        z_scores = zscore(df_copy[existing_numeric])
        is_not_outlier = (np.abs(z_scores) < self.outlier_threshold).all(axis=1)
        original_rows = len(df_copy)
        df_clean = df_copy[is_not_outlier]
        removed_rows = original_rows - len(df_clean)
        if removed_rows > 0:
            logger.info(f"Eliminados {removed_rows} outliers.")
        return df_clean

    def fit(self, df: pd.DataFrame):
        """
        Ajusta el pipeline de preprocesamiento a los datos.
        """
        logger.info("Ajustando el preprocesador de datos...")
        df_eng = self._create_features(df.drop(columns=['PatientID'], errors='ignore'))
        
        self.all_numeric_features_ = [f for f in self.numeric_features + ['Age_BMI_Interaction', 'TumorSize_SurvivalRatio'] if f in df_eng.columns]
        self.all_categorical_features_ = [f for f in self.categorical_features + ['BMI_Category'] if f in df_eng.columns]

        df_eng[self.all_numeric_features_] = df_eng[self.all_numeric_features_].fillna(df_eng[self.all_numeric_features_].mean())
        df_eng[self.all_categorical_features_] = df_eng[self.all_categorical_features_].fillna('Unknown')

        numeric_transformer = Pipeline(steps=[('scaler', RobustScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])

        self._preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.all_numeric_features_),
                ('cat', categorical_transformer, self.all_categorical_features_)
            ],
            remainder='passthrough'
        )
        
        self._preprocessor.fit(df_eng)
        logger.info("Preprocesador ajustado exitosamente.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Aplica el pipeline de preprocesamiento ya ajustado a los datos.
        """
        if self._preprocessor is None:
            raise RuntimeError("El preprocesador no ha sido ajustado. Llama a 'fit' primero.")
        
        logger.info("Transformando datos...")
        df_eng = self._create_features(df.drop(columns=['PatientID'], errors='ignore'))
        
        df_eng[self.all_numeric_features_] = df_eng[self.all_numeric_features_].fillna(df_eng[self.all_numeric_features_].mean())
        df_eng[self.all_categorical_features_] = df_eng[self.all_categorical_features_].fillna('Unknown')

        df_clean = self._handle_outliers(df_eng)
        
        X_processed = self._preprocessor.transform(df_clean)
        logger.info("Datos transformados exitosamente.")
        
        return X_processed, df_clean
