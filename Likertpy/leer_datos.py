import pandas as pd
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
            return self.readers[self.file_extension](self.crear_path())
        except Exception as e:
            raise RuntimeError(f"Error al leer el archivo '{self.file_path}': {e}")


    def crear_path(self) -> Path:
        try:
            return Path(self.folder, self.file)
        except Exception as err:
            print(f"Error al crear path. {err}")
            raise Exception(err)