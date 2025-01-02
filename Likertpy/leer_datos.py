import pandas as pd
import numpy as np
from pathlib import Path
import Likertpy.scales


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

    def __init__(self, folder: str, file: str):
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

    def __init__(self, data: pd.DataFrame, group: str):
        self.data = data
        self.group = group

    def clean_data(self):
        """
        Cleans and preprocesses survey data for analysis.

        This function filters, cleans, and processes the survey data by:
        - Selecting a specific group of interest (e.g., G1).
        - Removing rows with missing values.
        - Filtering responses for a specific survey iteration.
        - Dropping unnecessary columns (e.g., survey number and folio).
        - Replacing numerical responses with corresponding string labels.

        Returns:
            pd.DataFrame: A cleaned DataFrame with processed survey data, ready for analysis.
        """

        # Select group of interest
        # In this case G1 but should be change to make it interactive
        data_MSAS_G = self.data.iloc[:, self._select_group_range()]

        # Drop Na Values
        data_MSAS_G = data_MSAS_G.dropna()

        # Select survey's number (0,1,2)
        data_MSAS_G = data_MSAS_G.loc[data_MSAS_G["('', 'numencuesta')"] == 0]
        data_MSAS_G = data_MSAS_G.drop("('', 'numencuesta')", axis=1)
        data_MSAS_G = data_MSAS_G.drop("('', 'folio')", axis=1)

        # Replace numerical data

        data_MSAS_G = self._replace_numerical_data(data_MSAS_G)

        return data_MSAS_G

    def _replace_numerical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces numerical data in a DataFrame with corresponding string labels.

        This method converts a DataFrame containing numerical responses into their respective
        string-based labels as defined in the `Likertpy.scales.msas` scale. It is designed
        for Likert-style datasets where numerical values represent specific answers.

        Args:
            data (pd.DataFrame): The input DataFrame with numerical values (floats) representing responses.

        Returns:
            pd.DataFrame: A DataFrame where numerical values have been replaced by their corresponding
            string labels.

        Notes:
            - The DataFrame is first converted to a string dtype to ensure compatibility.
            - The mapping of numerical values to labels is based on the `Likertpy.scales.msas` attribute.
        """

        # Change dataframe's dtype from float to string (object)
        data = data.astype("str")
        # For every answer in selected scale
        for response in range(len(Likertpy.scales.msas_G1)):
            # For each column
            for element in data.columns:
                # Change numerical data for question' answer
                data.loc[data[element] == str(response) + ".0", element] = (
                    Likertpy.scales.msas_G1[response]
                )
        return data

    def _select_group_range(self):
        # Dict of ranges
        ranges = {"G1": np.r_[0:24]}
        # Select group range
        try:
            group_range = ranges[self.group]
        except KeyError:
            raise ValueError(f"Group '{self.group}' not found in questions ranges")
        return group_range
