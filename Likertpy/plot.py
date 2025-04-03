"""
Plot Likert-style data and Heatmaps from Pandas using Matplotlib

Initially based on code from Austin Cory Bart
https://stackoverflow.com/a/41384812

Base code from nmalkin
https://github.com/nmalkin/plot-likert


Note:
the data must be strings
for a float: scores.applymap(int).applymap(str)
"""

import logging
import typing
from warnings import warn
from textwrap import wrap

import numpy as np
import pandas as pd


try:
    import matplotlib.axes
    import matplotlib.pyplot as plt
except RuntimeError as err:
    logging.error(
        "Couldn't import matplotlib, likely because this package is running in an environment that doesn't support it (i.e., without a graphical output). See error for more information."
    )
    raise err

from Likertpy.utils import *
import Likertpy.colors as builtin_colors
from Likertpy import Interval, FileRead, cleanData, Scale

HIDE_EXCESSIVE_TICK_LABELS = True
PADDING_LEFT = 0.02  # fraction of the total width to use as padding
PADDING_RIGHT = 0.04  # fraction of the total width to use as padding
BAR_LABEL_FORMAT = (
    "%g"  # if showing labels, how should the number be formatted? e.g., "%.2g"
)
BAR_LABEL_SIZE_CUTOFF = 0.05


class PlotLikertError(ValueError):
    pass

class HeatMapError(ValueError):
    pass

class ConfigurePlot:
    """
    Handles the layout and configuration of Likert-scale survey plots.

    This class provides utility functions for centering response data, adjusting x-axis labels,
    and formatting bar labels in a stacked horizontal bar chart. It ensures that responses
    are visually aligned around a neutral point, and optimizes the plot for readability.

    Methods
    -------
    _configure_rows(scale, counts)
        Centers response counts around the neutral category and pads rows for balanced visualization.

    _set_x_labels(padded_counts, xtick_interval, axes, center, counts)
        Computes and applies x-axis labels, ensuring even distribution and visibility constraints.

    _set_bar_labels(axes, compute_percentages, counts_sum, bar_labels_color, scale)
        Configures and applies labels to bar segments while ensuring readability based on bar size.
    """

    def __init__(self):
        pass

    def _configure_rows(
        self, scale, counts: pd.DataFrame
    ) -> tuple[pd.DataFrame, float, pd.DataFrame]:
        """
        Centers rows of a DataFrame around the Neutral response for Likert-style data.

        Pads each row based on the cumulative counts to the left of the Neutral response,
        ensuring alignment for balanced visualizations. The rows are reversed to maintain
        the original order of questions in the plot.

        Args:
            scale (list): Likert scale values (e.g., [1, 2, 3, 4, 5]).
            counts (pd.DataFrame): Response counts for each category (rows are questions,
                columns are scale values).

        Returns:
            tuple:
                - pd.DataFrame: The adjusted DataFrame with padded and reversed rows.
                - float: The maximum center value used for padding.
                - pd.DataFrame: The full padded DataFrame with the added padding column.

        Notes:
            - For even-length scales, the Neutral category is split evenly.
            - Padding column is excluded from legends by renaming.
        """

        # Pad each row/question from the left, so that they're centered around the middle (Neutral) response
        scale_middle = len(scale) // 2

        if scale_middle == len(scale) / 2:
            middles = counts.iloc[:, 0:scale_middle].sum(axis=1)
        else:
            middles = (
                counts.iloc[:, 0:scale_middle].sum(axis=1)
                + counts.iloc[:, scale_middle] / 2
            )

        center = middles.max()

        padding_values = (middles - center).abs()
        padded_counts = pd.concat([padding_values, counts], axis=1)
        # Hide the padding row from the legend
        padded_counts = padded_counts.rename({0: ""}, axis=1)

        # Reverse rows to keep the questions in order
        # (Otherwise, the plot function shows the last one at the top.)
        rows = padded_counts.iloc[::-1]
        return rows, center, padded_counts

    def _set_x_labels(
        self,
        padded_counts: pd.DataFrame,
        xtick_interval: int,
        axes: matplotlib.axes.Axes,
        center: float,
        counts: pd.DataFrame,
    ):
        """
        Computes and sets the x-axis labels and their positions for a Likert-style plot.

        The function calculates appropriate x-axis tick positions and labels, ensuring
        balance around the center (Neutral response) and avoiding excessive labels
        beyond the total count or percentage.

        Args:
            padded_counts (pd.DataFrame): DataFrame with padded and centered counts.
            xtick_interval (int): Desired interval between x-ticks. If `None`,
                it is calculated dynamically based on axis width and tick space.
            axes (matplotlib.axes.Axes): The matplotlib axes object for the plot.
            center (float): The value used as the central reference point for padding.
            counts (pd.DataFrame): Original response counts for validation and scaling.

        Returns:
            tuple:
                - xvalues (np.ndarray): Array of x-tick positions.
                - xlabels (list): List of corresponding x-axis labels.

        Notes:
            - Dynamically adjusts tick interval using the `Interval` class if not provided.
            - Ensures tick labels do not exceed the maximum count or 100% when `HIDE_EXCESSIVE_TICK_LABELS` is enabled.
            - Includes both left (negative direction) and right (positive direction) labels centered around `center`.
        """

        # Compute and show x labels
        max_width = int(round(padded_counts.sum(axis=1).max()))
        if xtick_interval is None:
            num_ticks = axes.xaxis.get_tick_space()
            interval_helper = Interval()
            interval = interval_helper.get_interval_for_scale(num_ticks, max_width)
        else:
            interval = xtick_interval

        right_edge = max_width - center
        right_labels = np.arange(interval, right_edge + interval, interval)
        right_values = center + right_labels
        left_labels = np.arange(0, center + 1, interval)
        left_values = center - left_labels
        xlabels = np.concatenate([left_labels, right_labels])
        xvalues = np.concatenate([left_values, right_values])

        xlabels = [int(l) for l in xlabels if round(l) == l]

        # Ensure tick labels don't exceed number of participants
        # (or, in the case of percentages, 100%) since that looks confusing
        if HIDE_EXCESSIVE_TICK_LABELS:
            # Labels for tick values that are too high are hidden,
            # but the tick mark itself remains displayed.
            total_max = counts.sum(axis="columns").max()
            xlabels = ["" if label > total_max else label for label in xlabels]

        return xvalues, xlabels

    def _set_bar_labels(
        self,
        axes: matplotlib.axes.Axes,
        compute_percentages: bool,
        counts_sum: pd.DataFrame,
        bar_labels_color,
        scale: list,
    ):
        """
        Configures and applies labels to the bars in a Likert-style plot.

        This function handles the placement, formatting, and visibility of bar labels, ensuring
        readability based on the bar size. It also verifies the compatibility of color settings
        with the given scale.

        Args:
            axes (matplotlib.axes.Axes): The matplotlib axes object containing the bar containers.
            compute_percentages (bool): Whether bar labels should display percentages or raw values.
            counts_sum (float): Total count of responses, used to determine the cutoff for small labels.
            bar_labels_color (str or list): Single color for all labels or a list of colors corresponding
                to each scale segment.
            scale (list): The Likert scale values, used to validate color assignments.

        Raises:
            PlotLikertError: If the number of colors in `bar_labels_color` does not match the scale length,
            or if matplotlib version is insufficient to render bar labels.

        Notes:
            - Labels below a size threshold (relative to `counts_sum` and `BAR_LABEL_SIZE_CUTOFF`) are hidden.
            - Requires matplotlib version 3.4.0 or higher for `bar_label` functionality.
        """

        bar_label_format = BAR_LABEL_FORMAT + ("%%" if compute_percentages else "")
        bar_size_cutoff = counts_sum * BAR_LABEL_SIZE_CUTOFF

        if isinstance(bar_labels_color, list):
            if len(bar_labels_color) != len(scale):
                raise PlotLikertError(
                    "list of bar label colors must have as many values as the scale"
                )
            bar_label_colors = bar_labels_color
        else:
            bar_label_colors = [bar_labels_color] * len(scale)

        for i, segment in enumerate(
            axes.containers[1:]  # the first container is the padding
        ):
            try:
                labels = axes.bar_label(
                    segment,
                    label_type="center",
                    fmt=bar_label_format,
                    padding=0,
                    color=bar_label_colors[i],
                    weight="bold",
                )
            except AttributeError:
                raise PlotLikertError(
                    "Rendering bar labels requires matplotlib version 3.4.0 or higher"
                )

            # Remove labels that don't fit because the bars are too small
            for label in labels:
                label_text = label.get_text()
                if compute_percentages:
                    label_text = label_text.rstrip("%")
                try:
                    number = float(label_text)
                except ValueError:
                    continue
                if number < bar_size_cutoff:
                    label.set_text("")


def plot_likert(
    df: typing.Union[pd.DataFrame, pd.Series, str],
    survey_number: int,
    group: str = "",
    format_scale: Scale = None,
    colors: builtin_colors.Colors = builtin_colors.default_msas,
    label_max_width: int = 30,
    drop_zeros: bool = False,
    figsize=(15, 9),
    xtick_interval: typing.Optional[int] = None,
    compute_percentages: bool = False,
    bar_labels: bool = False,
    clean_data: bool = False,
    bar_labels_color: typing.Union[str, typing.List[str]] = "white",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Generates a horizontal bar chart to visualize Likert-scale survey responses.

    This function processes Likert-type survey data and creates a stacked horizontal
    bar chart representing the distribution of responses. It supports various customization
    options, including different response scales, color schemes, data formatting, and display
    adjustments.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        A DataFrame with survey questions as column names and Likert-scale responses
        (typically 0-4 or 1-5) as cell values.
    group : str
        The column name indicating the group to which the responses belong.
    survey_number : int
        The identifier for the survey iteration or dataset version being analyzed.
    plot_scale : list
        The scale used for the actual plot: a list of strings representing Likert-scale
        answer options in the correct order.
    format_scale : list of str, optional
        An optional list defining how response values should be formatted. If responses are
        numeric, this can replace them with text labels. If NA values exist, the list should
        start with a corresponding empty or zero value.
    colors : list of str, default = builtin_colors.default_msas
        A list of colors (hex codes or RGB tuples) for plotting. If colors do not render
        correctly, consider appending a transparent color (`"#ffffff00"`) at the beginning.
    label_max_width : int, default = 30
        Maximum character width for wrapping question labels on the y-axis.
    drop_zeros : bool, default = False
        Whether to remove NA values from the dataset before plotting.
    figsize : tuple (int, int), optional
        A tuple `(width, height)` controlling the size of the final figure.
    xtick_interval : int, optional
        Interval between x-axis tick labels.
    compute_percentages : bool, default = False
        Whether to convert response counts into percentages before plotting.
    bar_labels : bool, default = False
        Whether to display numerical labels inside the bar segments.
    clean_data : bool, default = True
        Whether to preprocess and clean the dataset before plotting.
    bar_labels_color : str or list of str, default = "white"
        Color(s) for the bar segment labels if `bar_labels=True`.
    **kwargs
        Additional keyword arguments passed to the pandas plotting function.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib Axes object containing the Likert-scale plot.

    Notes
    -----
    - If `format_scale` is provided, response values will be reformatted accordingly.
    - If `compute_percentages=True`, counts will be normalized and displayed as percentages.
    - A vertical dashed line is drawn at the center to highlight neutral responses.
    - The legend automatically adjusts based on the provided scale.
    - Padding adjustments are applied to optimize spacing for visual clarity.
    """

    conf_plot = ConfigurePlot()
    # If a string is passed, read the file
    if isinstance(df, str):
        data = FileRead(folder="IN", file=df).read_file_to_dataframe()
        # print(data.head())
        df_cleaned, plot_scale = cleanData(
            data, group=group, file_name=df, survey_number=survey_number
        ).clean_data()
    # If needed, clean the data
    if clean_data:
        df_cleaned, plot_scale = cleanData(
            df, group=group, survey_number=survey_number
        ).clean_data()

    # Format column names for better readability
    df_cleaned = clean_column_names(select_survey_name(df),df_cleaned)

    if format_scale:
        df_fixed = likert_response(df_cleaned, format_scale)
    else:
        df_fixed = df_cleaned
        format_scale = plot_scale

    counts = likert_counts(df_fixed, format_scale, label_max_width, drop_zeros)

    if drop_zeros:
        plot_scale = plot_scale[1:]

    # Re-compute counts as percentages, if requested
    if compute_percentages:
        counts = _compute_counts_percentage(counts)
        counts_are_percentages = True
    else:
        counts_are_percentages = False
    # configure the rows to plot likert
    final_rows, center, padded_counts = conf_plot._configure_rows(plot_scale, counts)

    # Start putting together the plot
    axes = final_rows.plot.barh(stacked=True, color=colors, figsize=figsize, **kwargs)

    # Draw center line
    center_line = axes.axvline(center, linestyle="--", color="black", alpha=0.5)
    center_line.set_zorder(-1)

    # Set x values and x labels for the plot ticks
    xvalues, xlabels = conf_plot._set_x_labels(
        padded_counts, xtick_interval, axes, center, counts
    )

    # Set xlabel
    if counts_are_percentages:
        xlabels = [str(label) + "%" if label != "" else "" for label in xlabels]
        axes.set_xlabel("Porcentaje de Respuestas", fontsize=20)
    else:
        axes.set_xlabel("Número de Respuestas", fontsize=20)

    axes.set_xticks(xvalues)
    axes.set_xticklabels(xlabels)

    # Reposition the legend if present
    if axes.get_legend():
        axes.legend(bbox_to_anchor=(1.05, 1), title="Escala de Respuestas")

    # Adjust padding
    counts_sum = counts.sum(axis="columns").max()
    # Pad the bars on the left (so there's a gap between the axis and the first section)
    padding_left = counts_sum * PADDING_LEFT
    # Tighten the padding on the right of the figure
    padding_right = counts_sum * PADDING_RIGHT
    x_min, x_max = axes.get_xlim()
    axes.set_xlim(x_min - padding_left, x_max - padding_right)

    # Add labels
    if bar_labels:
        conf_plot._set_bar_labels(
            axes, compute_percentages, counts_sum, bar_labels_color, plot_scale
        )

    # Add name
    if isinstance(df, str):
        plot_name = select_survey_name(
            df
        )  # Select the survey name based on df, it only works for df = str
    else:
        plot_name = "Survey"
    axes.set_title(plot_name.upper(), fontsize=30)
    plt.show()
    return axes


def plot_mode(
    df: typing.Union[str, pd.DataFrame, pd.Series], group: str = None, **kwargs
) -> matplotlib.axes.Axes:
    """
    Generates a heatmap representing the mode of survey responses for a given group.

    This function processes the provided survey data by:
    - Cleaning and structuring the data for heatmap visualization.
    - Computing the mode of responses across three survey iterations.
    - Converting the data to float format for proper visualization.
    - Creating and displaying a heatmap using `matplotlib`.

    Args:
        df (Union[pd.DataFrame, pd.Series]): The survey data containing responses.
        group (str): The group identifier to filter and process the data.
        **kwargs: Additional keyword arguments for future customization.

    Returns:
        matplotlib.axes.Axes: The axes object of the generated heatmap.

    Raises:
        ValueError: If `df` is empty or not a valid DataFrame/Series.
        TypeError: If `group` is not a string.
        KeyError: If the required group is not found in the dataset.
    """

    # Validate input types
    if not isinstance(df, (str, pd.DataFrame, pd.Series)):
        raise ValueError(
            "The 'df' argument must be a pandas DataFrame or Path to File."
        )
    if isinstance(df, (pd.DataFrame, pd.Series)):
        if df.empty:
            raise ValueError("The provided dataset is empty. Cannot compute mode.")
    if not isinstance(group, str):
        if group is None and "apca" in df:
            pass
        else:
            raise TypeError("The 'group' argument must be a string.")

    if isinstance(df, str):
        data = FileRead(folder="IN", file=df).read_file_to_dataframe()
    # Clean and parse data
    heathmap_data = _configure_data_for_heatmap(fileName=df, data=data, group=group)

    # calculate mode
    mode_df = calculate_mode(heathmap_data)

    # change dtype to float
    mode_df = mode_df.astype(float)

    # Create the heatmap
    axes = _create_heatmap(mode_df, "Moda", group, fileName=df)
    return axes


def plot_max(
    df: typing.Union[str,pd.DataFrame, pd.Series], group: str, **kwargs
) -> matplotlib.axes.Axes:
    """
    Plots a heatmap of the maximum values across three cleaned DataFrames.

    This function cleans the input survey data, calculates the element-wise maximum
    across three DataFrames, and then generates a heatmap visualization. The heatmap
    displays the maximum value at each position across the three DataFrames.

    Args:
        df (Union[pd.DataFrame, pd.Series]): The input DataFrame or Series containing
            the survey data.
        group (str): The group identifier (e.g., "G1", "G2", "G3") to filter the data
            for the specific survey group.
        **kwargs: Additional keyword arguments passed to the heatmap plotting function.

    Returns:
        matplotlib.axes.Axes: The axes object for the generated heatmap.

    Raises:
        ValueError: If the `df` argument is neither a pandas DataFrame nor a Series,
            or if the provided dataset is empty.
        TypeError: If the `group` argument is not a string.

    Example:
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> plot_max(df, group="G1")
    """
    # Validate input types
    if not isinstance(df, (str, pd.DataFrame, pd.Series)):
        raise HeatMapError("The 'df' argument must be a pandas DataFrame or Series.")
    if isinstance(df, (pd.DataFrame, pd.Series)):
        if df.empty:
            raise HeatMapError("The provided dataset is empty. Cannot compute mode.")
    if not isinstance(group, str):
        raise TypeError("The 'group' argument must be a string.")

    if isinstance(df, str):
        data = FileRead(folder="IN", file=df).read_file_to_dataframe()
    # Clean and parse data
    heathmap_data = _configure_data_for_heatmap(fileName=df, data=data, group=group)

    # calculate maximum
    max_df = calculate_max(heathmap_data)

    # change dtype to float
    max_df = max_df.astype(float)

    # Create the heatmap
    axes = _create_heatmap(max_df, "Máximo", group, fileName=df)
    return axes


def plot_min(
    df: typing.Union[str, pd.DataFrame, pd.Series], group: str, **kwargs
) -> matplotlib.axes.Axes:
    """
    Generates a heatmap representing the minimum values across multiple datasets.

    This function processes the given DataFrame or Series, extracts relevant data for
    heatmap visualization, computes the element-wise minimum across datasets, and
    plots the resulting heatmap.

    Args:
        df (Union[pd.DataFrame, pd.Series]): A pandas DataFrame or Series containing
            the data to be analyzed.
        group (str): The name of the group (e.g., "G1", "G2", "G3") used to filter and preprocess the data.
        **kwargs: Additional keyword arguments for potential customization.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object containing the heatmap.

    Raises:
        ValueError: If `df` is not a DataFrame or Series, or if it is empty.
        TypeError: If `group` is not a string.

    Notes:
        - The function internally cleans and transposes the data before computing the minimum.
        - The minimum is calculated across three preprocessed datasets.
        - The resulting values are converted to `float` to ensure consistency in visualization.
        - Uses a predefined color scheme (`coolwarm`) for heatmap generation.
    """
    # Validate input types
    if not isinstance(df, (str, pd.DataFrame, pd.Series)):
        raise ValueError("The 'df' argument must be a pandas DataFrame or Series.")
    if isinstance(df, (pd.DataFrame, pd.Series)):
        if df.empty:
            raise ValueError("The provided dataset is empty. Cannot compute mode.")
    if not isinstance(group, str):
        raise TypeError("The 'group' argument must be a string.")

    if isinstance(df, str):
        data = FileRead(folder="IN", file=df).read_file_to_dataframe()
    # Clean and parse data
    heathmap_data = _configure_data_for_heatmap(fileName=df, data=data, group=group)

    # calculate minimum
    min_df = calculate_min(heathmap_data)

    # change dtype to float
    min_df = min_df.astype(float)

    # Create the heatmap
    axes = _create_heatmap(min_df, "Mínimo", group, fileName=df)
    return axes


def plot_gradient(
    df: typing.Union[str, pd.DataFrame, pd.Series], group: str, **kwargs
) -> matplotlib.axes.Axes:
    # Validate input types
    if not isinstance(df, (str, pd.DataFrame, pd.Series)):
        raise ValueError("The 'df' argument must be a pandas DataFrame or Series.")
    if isinstance(df, (pd.DataFrame, pd.Series)):
        if df.empty:
            raise ValueError("The provided dataset is empty. Cannot compute mode.")
    if not isinstance(group, str):
        raise TypeError("The 'group' argument must be a string.")

    if isinstance(df, str):
        data = FileRead(folder="IN", file=df).read_file_to_dataframe()
    # Clean and parse data
    heathmap_data = _configure_data_for_heatmap(fileName=df, data=data, group=group)

    # calculate gradient
    gradient_df = calculate_gradient(heathmap_data)

    # create the heatmap
    axes = _create_heatmap(gradient_df, "Gradiente", group, fileName=df)
    return axes


def likert_counts(
    df: typing.Union[pd.DataFrame, pd.Series],
    scale: Scale,
    label_max_width=30,
    drop_zeros=False,
) -> pd.DataFrame:
    """
    Computes the count of each response category in a Likert-style dataset.

    This function validates the responses against the provided Likert scale, counts the occurrences
    of each response, and reformats long question labels for improved readability in plots.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        A dataset containing Likert-style responses. Column names represent questions, and
        cell values represent the responses.
    scale : list
        The Likert scale used for validation. This should be a list of possible response values,
        ordered from the lowest to the highest.
    label_max_width : int, optional (default=30)
        The maximum character width for question labels before wrapping text for better
        readability in visualizations.
    drop_zeros : bool, optional (default=False)
        If True, drops columns where the response count is zero (e.g., removing "0" values
        if the scale includes it).

    Returns
    -------
    pandas.DataFrame
        A DataFrame where rows represent questions and columns correspond to the response
        categories in the given Likert scale, containing the count of each response.

    Raises
    ------
    PlotLikertError
        If a response in the dataset does not match any value in the provided Likert scale,
        indicating potential issues such as extra whitespace, incorrect capitalization,
        or mismatched data types (e.g., int vs. str).

    Notes
    -----
    - The function ensures compatibility with different versions of Pandas (e.g., handling `.map()` vs. `.applymap()`).
    - It replaces long question labels with wrapped versions for better visualization.
    - It correctly aligns the response counts to match the order of the given Likert scale.
    """
    if type(df) == pd.core.series.Series:
        df = df.to_frame()

    def validate(value):
        if (not pd.isna(value)) and (value not in scale):
            raise PlotLikertError(
                f"A response was found with value `{value}`, which is not one of the values in the provided scale: {scale}. If this is unexpected, you might want to double-check for extra whitespace, capitalization, spelling, or type (int versus str)."
            )

    try:
        df.map(validate)
    except AttributeError:  # for compatibility with Pandas < 2.1.0
        df.applymap(validate)

    # fix long questions for printing
    old_labels = list(df)
    new_labels = ["\n".join(wrap(str(l), label_max_width)) for l in old_labels]
    if pd.__version__ >= "1.5.0":
        df = df.set_axis(new_labels, axis=1, copy=True)
    else:
        df = df.set_axis(new_labels, axis=1, inplace=False)

    counts_unordered = df.apply(lambda row: row.value_counts())
    counts = counts_unordered.reindex(scale).T
    counts = counts.fillna(0)

    # remove NA scores
    if drop_zeros == True:
        counts = counts.drop("0", axis=1)

    return counts


def likert_percentages(
    df: pd.DataFrame, scale: Scale, width=30, zero=False
) -> pd.DataFrame:
    """
    Given a dataframe of Likert-style responses, returns a new one
    reporting the percentage of respondents that chose each response.
    Percentages are rounded to integers.
    """

    counts = likert_counts(df, scale, width, zero)

    # Warn if the rows have different counts
    # If they do, the percentages shouldn't be compared.
    responses_per_question = counts.sum(axis=1)
    responses_to_first_question = responses_per_question.iloc[0]
    responses_same = responses_per_question == responses_to_first_question
    if not responses_same.all():
        warn(
            "In your data, not all questions have the same number of responses. i.e., different numbers of people answered each question. Therefore, the percentages aren't directly comparable: X% for one question represents a different number of responses than X% for another question, yet they will appear the same in the percentage graph. This may be misleading to your reader."
        )

    try:
        return counts.apply(lambda row: row / row.sum(), axis=1).map(lambda v: 100 * v)
    except AttributeError:  # for compatibility with Pandas < 2.1.0
        return counts.apply(lambda row: row / row.sum(), axis=1).applymap(
            lambda v: 100 * v
        )


def _compute_counts_percentage(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of response counts, return a new one
    with the response counts converted to percentages.
    """
    # Warn if the rows have different counts
    # If they do, the percentages shouldn't be compared.
    responses_per_question = counts.sum(axis="columns")
    responses_to_first_question = responses_per_question.iloc[0]
    responses_same = responses_per_question == responses_to_first_question
    if not responses_same.all():
        warn(
            "In your data, not all questions have the same number of responses. i.e., different numbers of people answered each question. Therefore, the percentages aren't directly comparable: X% for one question represents a different number of responses than X% for another question, yet they will appear the same in the percentage graph. This may be misleading to your reader."
        )
    return counts.divide(counts.sum(axis="columns"), axis="rows") * 100


def likert_response(df: pd.DataFrame, scale: Scale) -> pd.DataFrame:
    """
    This function replaces values in the original dataset to match one of the plot_likert
    scales in scales.py. Note that you should use a '_0' scale if there are NA values in the
    orginal data.
    """
    for i in range(0, len(scale)):
        try:
            df = df.map(lambda x: scale[i] if str(i) in x else x)
        except AttributeError:  # for compatibility with Pandas < 2.1.0
            df = df.map(lambda x: scale[i] if str(i) in x else x)
    return df


def raw_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    The purpose of this function is to determine the scale(s) used in the dataset.
    """
    df_m = df.melt()
    scale = df_m["value"].drop_duplicates()
    return scale


def _configure_data_for_heatmap(fileName: str, data: pd.DataFrame, group: str):
    """
    Prepares and structures survey data for heatmap visualization.

    This function processes survey data by:
    - Cleaning and filtering responses for a specified group (`group`) across three survey iterations (0, 1, 2).
    - Resetting the index of each cleaned dataset for consistency.
    - Transposing the cleaned data to facilitate heatmap generation.

    Args:
        df (pd.DataFrame): The raw survey dataset containing responses for multiple groups.
        group (str): The group identifier (e.g., "G1", "G2", etc.) used to filter relevant survey questions.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        A tuple containing three transposed DataFrames, each representing a survey iteration (0, 1, 2).

    Raises:
        ValueError: If the specified group is not found in the dataset.
        KeyError: If required columns for the survey responses are missing.

    Notes:
        - The `CleanData` class is used to clean the data without replacing numerical responses.
        - The transposed format is useful for heatmap visualizations, where questions are typically placed as rows.

    """
    # Verify data
    valid_groups = {"G1", "G2", "G3", "G4", "G5"}  # Se debe actualizar según corresponda
    if group not in valid_groups:
        if group is None and "apca" in fileName:
            pass
        else:
            raise ValueError(
                f"Invalid group '{group}'. Expected one of {valid_groups}."
            )
    
    try:
        # Data cleaning
        data_cleaned_1 = cleanData(
            data,
            group=group,
            file_name=fileName,
            survey_number=0,
            replace_numerical_data=False,
            convert_to_numerical=True,
        ).clean_data()[0]

        # Format column names for better readability
        data_cleaned_1 = clean_column_names(select_survey_name(fileName),data_cleaned_1)
        
        data_cleaned_2 = cleanData(
            data,
            group=group,
            file_name=fileName,
            survey_number=1,
            replace_numerical_data=False,
            convert_to_numerical=True,
        ).clean_data()[0]
        # Format column names for better readability
        data_cleaned_2 = clean_column_names(select_survey_name(fileName),data_cleaned_2)

        data_cleaned_3 = cleanData(
            data,
            group=group,
            file_name=fileName,
            survey_number=2,
            replace_numerical_data=False,
            convert_to_numerical=True,
        ).clean_data()[0]

        # Format column names for better readability
        data_cleaned_3 = clean_column_names(select_survey_name(fileName),data_cleaned_3)

    except KeyError as e:
        raise KeyError(f"Column access error while processing group '{group}': {e}")

    except ValueError as e:
        raise ValueError(f"Error while cleaning data for group '{group}': {e}")

    # Verufy that DataFrames are not emptys after cleaning
    if data_cleaned_1.empty or data_cleaned_2.empty or data_cleaned_3.empty:
        raise ValueError(
            f"Cleaned data for group '{group}' is empty. Check the dataset."
        )

    try:
        # Reset index
        data_cleaned_1 = data_cleaned_1.reset_index(drop=True)
        data_cleaned_2 = data_cleaned_2.reset_index(drop=True)
        data_cleaned_3 = data_cleaned_3.reset_index(drop=True)

        # Transpose data
        data_transpose_1 = data_cleaned_1.transpose()
        data_transpose_2 = data_cleaned_2.transpose()
        data_transpose_3 = data_cleaned_3.transpose()

    except KeyError as e:
        raise KeyError(f"Unexpected KeyError while restructuring data: {e}")

    return data_transpose_1, data_transpose_2, data_transpose_3


def _create_heatmap(
    data: pd.DataFrame, type: str, group: str, fileName: str, **kwargs
) -> matplotlib.axes.Axes:
    """
    Generates a heatmap to visualize survey data distributions.

    This function creates a heatmap using `matplotlib` to display the values of a given
    DataFrame, using a color gradient to represent variations in the data.

    Args:
        data (pd.DataFrame): The DataFrame containing the numerical values to be plotted.
        type (str): The type or category of the heatmap, must be "Moda", "Maximo", "Minimo", Gradiente.
        group (str): The group identifier to be included in the plot title.
        **kwargs: Additional keyword arguments for future customization.

    Returns:
        matplotlib.axes.Axes: The axes object of the generated heatmap.

    Raises:
        ValueError: If `data` is empty or not a valid DataFrame.
        TypeError: If `type` or `group` are not strings.
    """
    # Validate input types
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' argument must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("The provided DataFrame is empty. Cannot generate heatmap.")
    if not isinstance(type, str) or not isinstance(group, str):
        if group is None and "apca" in fileName:
            pass
        else:
            raise TypeError("The 'type' and 'group' arguments must be strings.")

    # initialize plot
    fig = plt.figure(figsize=(15, 9))
    axes = fig.gca()

    # plot mode
    im = axes.imshow(data, cmap="coolwarm", aspect="auto")

    # Add colorbar
    fig.colorbar(im, ax=axes)

    # Select name
    if isinstance(fileName, str):
        plot_name = select_survey_name(
            fileName
        ).upper()  # Select the survey name based on df, it only works for df = str
    else:
        plot_name = "Survey"

    # Add titles and labels
    axes.set_yticks(range(len(data.index)))
    axes.set_yticklabels(data.index)
    if group:
        axes.set_title(f"{type} {plot_name} {group}", y=1.08, fontsize=30)
    else:
        axes.set_title(f"{type} {plot_name}", y=1.08, fontsize=30)
    axes.set_ylabel("Preguntas", fontsize=15)
    axes.set_xlabel("Pacientes", fontsize=15)

    plt.show()

    return axes
