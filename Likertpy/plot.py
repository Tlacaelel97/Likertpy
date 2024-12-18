"""
Plot Likert-style data from Pandas using Matplotlib

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

from Likertpy import Scale
import Likertpy.colors as builtin_colors
from Likertpy import Interval

HIDE_EXCESSIVE_TICK_LABELS = True
PADDING_LEFT = 0.02  # fraction of the total width to use as padding
PADDING_RIGHT = 0.04  # fraction of the total width to use as padding
BAR_LABEL_FORMAT = (
    "%g"  # if showing labels, how should the number be formatted? e.g., "%.2g"
)
BAR_LABEL_SIZE_CUTOFF = 0.05


class PlotLikertError(ValueError):
    pass


class ConfigurePlot:

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
    df: typing.Union[pd.DataFrame, pd.Series],
    plot_scale: Scale,
    format_scale: Scale = None,
    colors: builtin_colors.Colors = builtin_colors.default_msas,
    label_max_width: int = 30,
    drop_zeros: bool = False,
    figsize=None,
    xtick_interval: typing.Optional[int] = None,
    compute_percentages: bool = False,
    bar_labels: bool = False,
    bar_labels_color: typing.Union[str, typing.List[str]] = "white",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot the given Likert-type dataset.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        A dataframe with questions in column names and answers recorded as cell values.
    plot_scale : list
        The scale used for the actual plot: a list of strings in order for answer options.
    format_scale : list of str
        Optional scale used to reformat the responses: \
        if your responses are numeric values, you can pass in this scale to replace them with text. \
        If your dataset has NA values, this list must have a corresponding 0/empty value at the beginning.
    colors : list of str
        A list of colors in hex string or RGB tuples to use for plotting. Attention: if your \
        colormap doesn't work right try appending transparent ("#ffffff00") in the first place.
    label_max_width : int
        The character wrap length of the y-axis labels.
    drop_zeros : bool
        Indicates whether the data have NA values that should be dropped (True) or not (False).
    figsize : tuple of (int, int)
        A tuple (width, heigth) that controls size of the final figure - \
        similarly to matplotlib
    xtick_interval : int
        Controls the interval between x-axis ticks.
    compute_percentages : bool, default = True,
        Convert the given response counts to percentages and display the counts as percentages in the plot.
    bar_labels : bool, default = False
        Show a label with the value of each bar segment on top of it
    bar_labels_color : str or list of str = "white",
        If showing bar labels, use this color (or colors) for the text
    **kwargs
        Options to pass to pandas plotting method.

    Returns
    -------
    matplotlib.axes.Axes
        Likert plot
    """
    conf_plot = ConfigurePlot()

    if format_scale:
        df_fixed = likert_response(df, format_scale)
    else:
        df_fixed = df
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
        axes.set_xlabel("Porcentaje de Respuestas")
    else:
        axes.set_xlabel("Número de Respuestas")

    axes.set_xticks(xvalues)
    axes.set_xticklabels(xlabels)

    # Reposition the legend if present
    if axes.get_legend():
        axes.legend(bbox_to_anchor=(1.05, 1))

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
    axes.set_title("MSAS")
    plt.show()
    return axes


def likert_counts(
    df: typing.Union[pd.DataFrame, pd.Series],
    scale: Scale,
    label_max_width=30,
    drop_zeros=False,
) -> pd.DataFrame:
    """
    Given a dataframe of Likert-style responses, returns a count of each response,
    validating them against the provided scale.
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
