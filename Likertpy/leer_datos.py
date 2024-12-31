import pandas as pd
import numpy as np
from pathlib import Path

class FileRead:
    """
    Read files and transform into pandas.DataFrame
    
    Attributes:
        lib (str) -- Name of the data folder
        file (str) -- Name of the data file
        path (Path) -- Path of the file that will be converted to pandas.DataFrame
        readers (dict) -- Different pandas file read methods
        file_extension (str) -- Identifyer of the file's extension
    """
    
    def __init__(self, folder:str, file:str):
        self.folder = folder
        self.file = file
        # Mapeo de extensiones a funciones de lectura
        self.readers = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".json": pd.read_json,
            ".parquet": pd.read_parquet,
        }
        
        # Identificar la extensión del archivo
        file_extension = file.rsplit('.', 1)[-1].lower()
        self.file_extension = f".{file_extension}"  # Aseguramos el formato con un punto


    def read_file_to_dataframe(self) -> pd.DataFrame:
        """
        Lee un archivo y lo convierte a un DataFrame de pandas según su extensión.

        Args:
            file_path (str): Ruta completa del archivo a leer.

        Returns:
            pd.DataFrame: DataFrame con los datos del archivo.

        Raises:
            ValueError: Si la extensión del archivo no es soportada.
        """
        # Verificar si la extensión está soportada
        if self.file_extension not in self.readers:
            raise ValueError(f"Formato de archivo no soportado: '{self.file_extension}'")

        # Usar la función correspondiente
        try:
            return self.readers[self.file_extension](self._crear_path())
        except Exception as e:
            raise RuntimeError(f"Error al leer el archivo '{self.file_path}': {e}")


    def _crear_path(self) -> Path:
        try:
            return Path(self.folder, self.file)
        except Exception as err:
            print(f"Error al crear path. {err}")
            raise Exception(err)
        
class CleanData:
    """
    Bad data could be:

    Empty cells
    Data in wrong format
    Wrong data
    Duplicates
    """

    def __init__(self, data:pd.DataFrame):
        self.data = data

    def clean_data(self):
        # Select group of interest
        # In this case G1 but should be change to make it interactive
        data_MSAS_G1 = self.data.iloc[:,np.r_[0:24]]
        return data_MSAS_G1

    

