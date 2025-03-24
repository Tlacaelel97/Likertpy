import pandas as pd
import numpy as np
from pathlib import Path
import Likertpy.scales

import typing


class FileRead:
    """
    A utility class for reading data files and converting them into a pandas DataFrame.

    This class supports reading files of various formats, including CSV, Excel, JSON, 
    and Parquet, by automatically detecting the file extension and using the appropriate 
    pandas reading method.

    Attributes
    ----------
    folder : str
        Name of the directory containing the data file.
    file : str
        Name of the data file to be read.
    readers : dict
        A mapping of supported file extensions to corresponding pandas read functions.
    file_extension : str
        The file extension (e.g., ".csv", ".xlsx", ".json" or ".parquet"), used to determine the appropriate reader method.

    Methods
    -------
    read_file_to_dataframe() -> pd.DataFrame
        Reads the specified file and returns its contents as a pandas DataFrame.
    _crear_path() -> Path
        Constructs and returns the full path of the file.
    
    Raises
    ------
    ValueError
        If the file extension is not supported.
    RuntimeError
        If an error occurs while reading the file.
    """
    def __init__(self, folder: str, file: str):
        self.folder = folder
        self.file = file
        # Mapeo de extensiones a funciones de lectura
        self.readers = {
            ".csv": lambda path: pd.read_csv(path, encoding='utf-8'),  
            ".xlsx": pd.read_excel,
            ".json": lambda path: pd.read_json(path, encoding='utf-8'),  
            ".parquet": pd.read_parquet,
        }

        # Identificar la extensión del archivo
        file_extension = file.rsplit(".", 1)[-1].lower()
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
            raise ValueError(
                f"Formato de archivo no soportado: '{self.file_extension}'"
            )

        # Usar la función correspondiente
        try:
            return self.readers[self.file_extension](self._crear_path())
        except Exception as e:
            raise RuntimeError(f"Error al leer el archivo '{self.file_path}': {e}")

    def _crear_path(self) -> Path:
        """
        Creates and returns a `Path` object by combining the directory and file name.

        This function uses the `Path` class from the `pathlib` module to construct 
        the full path of a file within a specified directory. If an error occurs while 
        creating the `Path`, it is caught and the original exception is re-raised.

        Returns
        -------
        Path
            A `Path` object representing the full file path.

        Raises
        ------
        Exception
            If an error occurs while creating the `Path`, an error message is printed, 
            and the original exception is re-raised.

        Notes
        -----
        - The function assumes that `self.folder` and `self.file` are correctly defined 
        as class attributes.
        - It is recommended to handle the exception at a higher level to avoid unexpected 
        program interruptions.
        """
        try:
            return Path(self.folder, self.file)
        except Exception as err:
            print(f"Error al crear path. {err}")
            raise Exception(err)

