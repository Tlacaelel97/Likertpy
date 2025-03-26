# Description: This module contains utility functions for processing Likert-scale data, and 
# to calculate mode, maximum, minimum and gradient for HeatMaps.

import pandas as pd
import numpy as np
import re

def select_survey_name(file_name: str) -> str:
    options = ["apca", "msas", "pedsql"]
    for option in options:
        if option.lower() in file_name.lower():
            return option

def calculate_mode(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """
    Computes the mode for each corresponding cell across three DataFrames.

    This function calculates the mode (most frequently occurring value) for each
    cell position across three given DataFrames. If multiple values share the highest
    frequency, the first occurring mode is selected. If no mode exists (i.e., all values
    are unique), NaN is assigned.

    Args:
        data (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): A tuple containing
            three DataFrames with identical structures (indices and columns).

    Returns:
        pd.DataFrame: A DataFrame with the same structure as the input DataFrames, where
        each cell contains the mode of the corresponding values from the three DataFrames.

    Raises:
        KeyError: If any of the DataFrames do not contain the expected indices or columns.
        IndexError: If there is no mode (i.e., all values are unique), resulting in an empty mode series.

    Notes:
        - Assumes that all input DataFrames have identical indices and columns.
        - Handles numerical and categorical data.
        - Uses `pandas.Series.mode()` to compute the most frequent value.
        - If no mode is found, the function assigns NaN to the corresponding cell.

    Example:
        >>> df1 = pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]})
        >>> df2 = pd.DataFrame({"A": [1, 3, 2], "B": [3, 5, 4]})
        >>> df3 = pd.DataFrame({"A": [1, 2, 3], "B": [3, 4, 6]})
        >>> calculate_mode((df1, df2, df3))
           A    B
        0  1.0  3.0
        1  NaN  4.0
        2  2.0  4.0
    """
    # Preparar un DataFrame vacío para almacenar los resultados
    mode_df = pd.DataFrame(index=data[0].index, columns=data[0].columns)

    # Calcular la moda para cada posición
    for col in data[0].columns:
        for idx in data[0].index:
            try:
                # Verificar que las claves existen en los DataFrames antes de acceder
                values = [
                    data[0].at[idx, col],
                    data[1].at[idx, col],
                    data[2].at[idx, col],
                ]
            except KeyError as e:
                raise KeyError(
                    f"Missing index '{idx}' or column '{col}' in one of the input DataFrames"
                ) from e

            # Calcular la moda de estos valores
            mode_value = pd.Series(values).mode()

            try:
                # Almacenar el primer valor de moda en el DataFrame (en caso de múltiples modas)
                mode_df.at[idx, col] = mode_value.iloc[0]
            except IndexError:
                # Si no hay moda (todos los valores son únicos), asignar NaN
                mode_df.at[idx, col] = np.nan

    return mode_df

def calculate_max(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculates the element-wise maximum across three DataFrames.

    This function computes the maximum value for each corresponding cell position
    across three given DataFrames. The result is a DataFrame where each cell contains
    the maximum value of the corresponding cells from the three input DataFrames.

    Args:
        data (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): A tuple containing three
            DataFrames with identical structures (indices and columns).

    Returns:
        pd.DataFrame: A DataFrame with the same structure as the input DataFrames, where
        each cell contains the maximum value of the corresponding cells from the three DataFrames.

    Example:
        >>> df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> df2 = pd.DataFrame({"A": [3, 2, 1], "B": [6, 5, 4]})
        >>> df3 = pd.DataFrame({"A": [2, 4, 3], "B": [5, 6, 7]})
        >>> calculate_max((df1, df2, df3))
           A  B
        0  3  6
        1  4  6
        2  3  7
    """
    if len(data) != 3:
        raise ValueError("The 'data' tuple must contain exactly three DataFrames.")
    max_data = pd.DataFrame(
        np.maximum(np.maximum(data[0], data[1]), data[2]), columns=data[2].columns
    )
    return max_data

def calculate_min(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """
    Computes the element-wise minimum across three DataFrames.

    This function takes a tuple containing three pandas DataFrames and returns a new
    DataFrame where each cell contains the minimum value from the corresponding
    cells of the three input DataFrames.

    Args:
        data (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): A tuple of three
            DataFrames with identical structure (same indices and columns).

    Returns:
        pd.DataFrame: A DataFrame containing the element-wise minimum values.

    Raises:
        ValueError: If the input tuple does not contain exactly three DataFrames.
        KeyError: If the DataFrames have misaligned indices or columns.

    Example:
        >>> df1 = pd.DataFrame({"A": [1, 5, 3], "B": [4, 2, 6]})
        >>> df2 = pd.DataFrame({"A": [2, 3, 1], "B": [5, 1, 7]})
        >>> df3 = pd.DataFrame({"A": [3, 4, 2], "B": [6, 0, 8]})
        >>> calculate_min((df1, df2, df3))
           A  B
        0  1  4
        1  3  0
        2  1  6
    """
    if len(data) != 3:
        raise ValueError("The 'data' tuple must contain exactly three DataFrames.")

    try:
        min_data = pd.DataFrame(
            np.minimum(np.minimum(data[0], data[1]), data[2]), columns=data[2].columns
        )
    except KeyError as e:
        raise KeyError("Mismatched indices or columns in input DataFrames.") from e

    return min_data

def calculate_gradient(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """
    Computes the gradient of change between successive DataFrames for Likert-scale data.

    This function calculates the element-wise gradient of change between three
    successive DataFrames, where the data represents responses on a Likert scale
    (values ranging from 0 to 4). The gradient is computed as the difference between
    consecutive DataFrames and averaged to obtain a single measure of change.

    Args:
        data (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): A tuple containing
            exactly three DataFrames representing successive states of a dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the averaged gradient of change,
        with the same structure as the input DataFrames.

    Raises:
        ValueError: If the input tuple does not contain exactly three DataFrames.

    Notes:
        - Since the input data represents Likert-scale responses (0 to 4), the gradient
          values will always range from -2 to 2:
            - The minimum possible gradient (-2) occurs when a response decreases by 2 points
              in consecutive steps (e.g., from 4 → 2 → 0).
            - The maximum possible gradient (2) occurs when a response increases by 2 points
              in consecutive steps (e.g., from 0 → 2 → 4).
            - A gradient of 0 indicates no overall change in the response trend.
        - The gradient is computed as:
          1. The difference between the second and first DataFrame (`data[1] - data[0]`).
          2. The difference between the third and second DataFrame (`data[2] - data[1]`).
          3. The final gradient is the mean of these two differences.
          4. To simplify the mathematical expresion, the mean gradient is calculated as
             `(data[2] - data[0]) / 2`.This is equivalent to the previous step-by-step
        - This approach smooths the changes over time, reducing noise in individual step differences.

    Example:
        >>> df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 3, 2]})
        >>> df2 = pd.DataFrame({"A": [2, 3, 4], "B": [3, 2, 1]})
        >>> df3 = pd.DataFrame({"A": [3, 4, 4], "B": [2, 1, 0]})
        >>> calculate_gradient((df1, df2, df3))
             A    B
        0  1.0 -1.0
        1  1.0 -1.0
        2  0.5 -1.0

    """
    if len(data) != 3:
        raise ValueError("The 'data' tuple must contain exactly three DataFrames.")

    # calculate the gradient as the mean of the differences between successive DataFrames
    mean_gradient = (data[2] - data[0]) / 2
    return mean_gradient

def clean_column_names(fileName:str,df:pd.DataFrame) -> pd.DataFrame:
    """

    """

    # Select and execute the appropriate cleaning function based on file name
    survey_type = select_survey_name(fileName)
    if survey_type == 'msas':
        return _clean_msas_column_names(df)
    elif survey_type == 'apca':
        return clean_apca()
    elif survey_type == 'pedsql':
        return clean_pedsql()
    else:
        raise ValueError(f"Unsupported survey type in filename: {self.file_name}")
        
def _clean_msas_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los nombres de las columnas de un DataFrame eliminando cualquier texto antes del primer número encontrado
    y removiendo el símbolo ']' si está al final del nombre de la columna.
    
    :param df: DataFrame de pandas con las columnas a limpiar.
    :return: DataFrame con los nombres de las columnas modificados.
    """
    def clean_name(col_name: str) -> str:
        match = re.search(r'\d+\)?\s*(.*)', col_name)
        cleaned_name = match.group(1).strip() if match else col_name
        return cleaned_name.rstrip(']')
    
    df = df.rename(columns={col: clean_name(col) for col in df.columns})
    return df