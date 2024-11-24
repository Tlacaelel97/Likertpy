import pandas as pd
from pathlib import Path

class FileRead:
    """
    Read files and transform into pandas.DataFrame
    
    Attributes:
        lib (str) -- Name of the data folder
        file (str) -- Name of the data file
        path (Path) -- Path of the file that will be converted to pandas.DataFrame
    """
    
    def __init__(self, lib:str, file:str):
        self.lib = lib
        self.file = file

    def leer_pdf(self) -> pd.DataFrame:
        try:
            # Abrir el archivo 
            df = pd.read_csv(self.crear_path())
            return df
        except Exception as err:
            print(f"Error al leer archivo. {err}")

    def crear_path(self) -> Path:
        try:
            return Path(self.lib, self.file)
        except Exception as err:
            print(f"Error al crear path. {err}")
            raise Exception(err)