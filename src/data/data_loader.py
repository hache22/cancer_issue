
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils import config
from src.utils.logger import logger

class DataLoader:
    """
    Clase responsable de la carga de datos desde la fuente definida
    en la configuración del proyecto.
    """
    def __init__(self, file_path: Path = config.RAW_DATA_FILE):
        """
        Inicializa el DataLoader.

        Args:
            file_path (Path): Ruta al archivo de datos. Por defecto, toma
                              el valor de la configuración central.
        """
        self.file_path = file_path

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Carga los datos desde el archivo CSV especificado.

        Returns:
            Optional[pd.DataFrame]: Un DataFrame de pandas con los datos cargados,
                                    o None si ocurre un error.
        """
        try:
            logger.info(f"Cargando datos desde: {self.file_path}")
            df = pd.read_csv(self.file_path)
            logger.info("Datos cargados exitosamente.")
            return df
        except FileNotFoundError:
            logger.error(f"Error: No se encontró el archivo en la ruta: {self.file_path}")
            return None
        except Exception as e:
            logger.error(f"Ha ocurrido un error inesperado al cargar los datos: {e}")
            return None
