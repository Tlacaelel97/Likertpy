import pandas as pd
import numpy as np

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
