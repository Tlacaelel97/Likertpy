import Likertpy.scales
from Likertpy.utils import select_survey_name

import pandas as pd
import numpy as np

class cleanData:
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
        file_name: str=None,
        survey_number: int = 0,
        replace_numerical_data: bool = True,
        convert_to_numerical: bool = False,
    ):
        self.data = data
        self.group = group
        self.file_name = file_name
        self.survey_number = survey_number
        self.replace = replace_numerical_data
        self.convert = convert_to_numerical

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

        # Select and execute the appropriate cleaning function based on file name
        survey_type = select_survey_name(self.file_name)
        if survey_type == 'msas':
            return self.clean_msas()
        elif survey_type == 'apca':
            return self.clean_apca()
        elif survey_type == 'pedsql':
            return self.clean_pedsql()
        else:
            raise ValueError(f"Unsupported survey type in filename: {self.file_name}")
        
    def clean_msas(self)->tuple:
        print("Cleaning MSAS data...")

        msas_range = self._msas_select_group_range() 
        # Select group of interest
        data_MSAS_G = self.data.iloc[:, msas_range]

        # Drop Na Values
        data_MSAS_G = data_MSAS_G.dropna()

        # Filter complete folios

        data_MSAS_G = self._filter_complete_folios(data_MSAS_G)

        # Select survey's number (0,1,2)
        data_MSAS_G = data_MSAS_G.loc[
            data_MSAS_G["numencuesta"] == self.survey_number
        ]
        data_MSAS_G = data_MSAS_G.drop("numencuesta", axis=1)
        data_MSAS_G = data_MSAS_G.drop("folio", axis=1)
        # # Replace numerical data
        if self.replace:
            data_MSAS_G = self._replace_numerical_data(data_MSAS_G)
        
        if self.convert:
            data_MSAS_G = self._convert_to_numerical(data_MSAS_G)

        
        return data_MSAS_G,self._scale

    def clean_apca(self)->tuple:
        print("Cleaning APCA data...")

        # Drop Na Values
        data_apca = self.data.dropna()

        # Filter complete folios

        data_apca_filtered = self._filter_complete_folios(data_apca)

        # Select survey's number (0,1,2)
        data_apca = data_apca_filtered.loc[
            data_apca_filtered["numencuesta"] == self.survey_number
        ]
        # Drop unnecessary columns
        data_apca = data_apca.drop("numencuesta", axis=1)
        data_apca = data_apca.drop("Forma de aplicación:", axis=1)
        data_apca = data_apca.drop("¿Quién responde las preguntas?", axis=1)
        data_apca = data_apca.drop("folio", axis=1)
        data_apca = data_apca.drop("Encuesta", axis=1)
        
        return data_apca, Likertpy.scales.apca

    def clean_pedsql(self)->tuple:
        print("Cleaning PedsQL data...")
        pedsql_range = self._pedsql_select_group_range()
        # Select group of interest
        data_pedsql = self.data.iloc[:, pedsql_range]
        # Drop Na Values
        data_pedsql = data_pedsql.dropna()
        # Filter complete folios
        data_pedsql = self._filter_complete_folios(data_pedsql)
        # Select survey's number (0,1,2)
        data_pedsql_G = data_pedsql.loc[
            data_pedsql["numencuesta"] == self.survey_number
        ]
        data_pedsql_G = data_pedsql_G.drop("numencuesta", axis=1)
        data_pedsql_G = data_pedsql_G.drop("folio", axis=1)
        data_pedsql_G = data_pedsql_G.drop('Unnamed: 0', axis=1)
        # print(data_pedsql.stack().unique().tolist())
        # print(data_pedsql_G.columns.to_list())
        # Replace numerical data
        if self.replace:
            data_pedsql_G = self._pedsql_replace_numerical_data(data_pedsql_G)
        
        if self.convert:
            data_pedsql_G = self._pedsql_convert_to_numerical(data_pedsql_G)
        
        return data_pedsql_G,self._scale

    def _msas_select_group_scale(self)->list:
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
        
    def _pedsql_select_group_scale(self)->list:
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
            "G1": Likertpy.scales.pedsql,
            "G2": Likertpy.scales.pedsql_2,
            "G3": Likertpy.scales.pedsql,
            "G4": Likertpy.scales.pedsql,
            "G5": Likertpy.scales.pedsql_5,
        }  # Define the scales for each group
        try:
            return scales[self.group]  # Return the scale for the selected group
        except KeyError:
            raise ValueError(f"Group '{self.group}' not found in scales")

    def _filter_complete_folios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra el dataframe para conservar solo los folios con exactamente 3 'numencuesta'.

        Args:
            df (pd.DataFrame): Dataframe con MultiIndex en columnas.

        Returns:
            pd.DataFrame: Dataframe filtrado.
        """
        # Identificar los folios únicos y contar los 'numencuesta' asociados
        folio_counts = df.groupby(("folio")).size()

        # Filtrar folios con exactamente 3 'numencuesta'
        valid_folios = folio_counts[folio_counts == 3].index
        filtered_df = df[df[("folio")].isin(valid_folios)]

        return filtered_df
    
    def _pedsql_replace_numerical_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
        self._scale = self._pedsql_select_group_scale()
        for response in range(len(self._scale)):
            # For each column
            for element in data.columns:
                # Change numerical data for question' answer
                data.loc[data[element] == str(response) + ".0", element] = self._scale[
                    response
                ]
        return data
    
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
        self._scale = self._msas_select_group_scale()
        for response in range(len(self._scale)):
            # For each column
            for element in data.columns:
                # Change numerical data for question' answer
                data.loc[data[element] == str(response) + ".0", element] = self._scale[
                    response
                ]
        return data

    def _convert_to_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts Likert-style responses in the DataFrame to numerical values (0 to 4).

        This method maps the string-based Likert responses in the DataFrame to their
        corresponding numerical values based on the group's scale.

        Args:
            data (pd.DataFrame): The input DataFrame containing Likert-style responses.

        Returns:
            pd.DataFrame: A DataFrame with responses converted to numerical values.

        Raises:
            ValueError: If a response in the DataFrame does not match the group's scale.
        """
        # Retrieve the scale for the current group
        self._scale = self._msas_select_group_scale()

        # Create a mapping from scale values to numerical indices
        scale_mapping = {value: idx for idx, value in enumerate(self._scale)}

        # Convert responses to numerical values
        try:
            data = data.map(lambda x: scale_mapping[x] if x in scale_mapping else x)
        except KeyError as e:
            raise ValueError(f"Unexpected value '{e.args[0]}' found in the data. Ensure all responses match the scale: {self._scale}")

        return data
    
    def _pedsql_convert_to_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts Likert-style responses in the DataFrame to numerical values (0 to 4).

        This method maps the string-based Likert responses in the DataFrame to their
        corresponding numerical values based on the group's scale.

        Args:
            data (pd.DataFrame): The input DataFrame containing Likert-style responses.

        Returns:
            pd.DataFrame: A DataFrame with responses converted to numerical values.

        Raises:
            ValueError: If a response in the DataFrame does not match the group's scale.
        """
        # Retrieve the scale for the current group
        self._scale = self._pedsql_select_group_scale()

        # Create a mapping from scale values to numerical indices
        scale_mapping = {value: idx for idx, value in enumerate(self._scale)}

        # Convert responses to numerical values
        try:
            data = data.map(lambda x: scale_mapping[x] if x in scale_mapping else x)
        except KeyError as e:
            raise ValueError(f"Unexpected value '{e.args[0]}' found in the data. Ensure all responses match the scale: {self._scale}")

        return data

    def _msas_select_group_range(self):
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
            "G4": np.r_[0, 1, 68:84],
        }
        # Select group range
        try:
            return ranges[self.group]
        except KeyError:
            raise ValueError(f"Group '{self.group}' not found in questions ranges")
        
    def _pedsql_select_group_range(self):
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
            "G1": np.r_[0:28],
            "G2": np.r_[0, 1, 2, 28:55],
            "G3": np.r_[0, 1, 2, 55:82],
            "G4": np.r_[0, 1, 2, 82:109],
            "G5": np.r_[0, 1, 2, 109:120,121:138,139:147],
        }
        # Select group range
        try:
            return ranges[self.group]
        except KeyError:
            raise ValueError(f"Group '{self.group}' not found in questions ranges")
        
    def cleaner(self):
        cleaners = {"apca":self.clean_apca,"msas":self.clean_msas,"pedsql":self.clean_pedsql}
        return cleaners[select_survey_name()]()
