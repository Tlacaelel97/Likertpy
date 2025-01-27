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
    A utility class for cleaning and preprocessing survey data.

    The `CleanData` class is designed to handle typical data issues such as:
    - Empty cells or missing values.
    - Data in an incorrect format.
    - Inconsistent or erroneous entries.
    - Duplicate records.

    This class provides methods to clean and format survey data for analysis by:
    - Selecting specific subsets of data based on predefined groups.
    - Replacing numerical responses with string labels using Likert scales.
    - Filtering rows based on survey iteration and removing unnecessary columns.

    Attributes:
        data (pd.DataFrame): The input DataFrame containing the raw survey data.
        group (str): The group identifier (e.g., "G1", "G2") for selecting specific data subsets.

    Methods:
        clean_data():
            Cleans and preprocesses survey data for the specified group.
        _replace_numerical_data(data: pd.DataFrame) -> pd.DataFrame:
            Replaces numerical responses in the data with their corresponding string labels.
        _select_group_scale() -> list:
            Retrieves the Likert scale associated with the specified group.
        _select_group_range() -> numpy.ndarray:
            Retrieves the range of question indices for the specified group.

    Example:
        >>> raw_data = pd.DataFrame(...)  # Load your survey data
        >>> cleaner = CleanData(raw_data, group="G1")
        >>> cleaned_data = cleaner.clean_data()
        >>> print(cleaned_data.head())

    Raises:
        ValueError: If the specified group is not supported or required columns are missing.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        group: str,
        survey_number: int = 0,
        replace_numerical_data: bool = True,
    ):
        self.data = data
        self.group = group
        self.survey_number = survey_number
        self.replace = replace_numerical_data

    def clean_data(self):
        """
        Cleans and preprocesses survey data for the selected group.

        This function prepares survey data for analysis by:
        - Selecting the range of questions corresponding to the group (`self.group`).
        - Removing rows with missing values.
        - Filtering responses for the first survey iteration (where survey number is 0).
        - Dropping unnecessary columns, such as survey number and folio.
        - Replacing numerical responses with their corresponding string labels based on the group's scale.

        Returns:
            pd.DataFrame: A cleaned and preprocessed DataFrame containing the survey data for the specified group.

        Raises:
            ValueError: If the specified group or required columns are missing.

        Example:
            >>> self.group = "G1"
            >>> cleaned_data = self.clean_data()
            >>> cleaned_data.head()
        """

        # Select group of interest
        data_MSAS_G = self.data.iloc[:, self._select_group_range()]

        # Drop Na Values
        data_MSAS_G = data_MSAS_G.dropna()

        # Select survey's number (0,1,2)
        data_MSAS_G = data_MSAS_G.loc[
            data_MSAS_G["('', 'numencuesta')"] == self.survey_number
        ]
        data_MSAS_G = data_MSAS_G.drop("('', 'numencuesta')", axis=1)
        data_MSAS_G = data_MSAS_G.drop("('', 'folio')", axis=1)

        # Replace numerical data
        if self.replace:
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
        msas_scale = self._select_group_scale()
        for response in range(len(msas_scale)):
            # For each column
            for element in data.columns:
                # Change numerical data for question' answer
                data.loc[data[element] == str(response) + ".0", element] = msas_scale[
                    response
                ]
        return data

    def _select_group_scale(self):
        """
        Selects the Likert scale corresponding to the specified group.

        This function retrieves the appropriate Likert scale based on the group
        attribute (`self.group`). Supported groups include "G1", "G2", and "G3".

        Returns:
            list: The Likert scale associated with the specified group.

        Raises:
            ValueError: If the specified group is not found in the predefined scales.

        """

        scales = {
            "G1": Likertpy.scales.msas_G1,
            "G2": Likertpy.scales.msas_G2,
            "G3": Likertpy.scales.msas_G3,
        }  # Define the scales for each group
        try:
            return scales[self.group]  # Return the scale for the selected group
        except KeyError:
            raise ValueError(f"Group '{self.group}' not found in scales")

    def _select_group_range(self):
        """
        Retrieves the range of question indices corresponding to the specified group.

        This function returns the appropriate range of indices for the selected group
        (`self.group`) to extract specific subsets of survey data.

        Returns:
            numpy.ndarray: The range of indices for the specified group.

        Raises:
            ValueError: If the specified group is not defined in the ranges dictionary.

        Notes:
            - Supported groups include "G1", "G2", "G3", and "G4".
            - The ranges correspond to specific sections of the survey data.
        """

        # Dict of ranges
        ranges = {
            "G1": np.r_[0:24],
            "G2": np.r_[0, 1, 24:46],
            "G3": np.r_[0, 1, 46:68],
            "G4": np.r_[0, 1, 68:79],
        }
        # Select group range
        try:
            return ranges[self.group]
        except KeyError:
            raise ValueError(f"Group '{self.group}' not found in questions ranges")
