from .leer_datos import FileRead
from .scales import Scale
from .interval import Interval
from .CleanData import cleanData

import Likertpy.plot as __internal__
from .plot import (
    plot_likert,
    plot_mode,
    plot_max,
    plot_min,
    plot_gradient,
    likert_counts,
    likert_percentages,
    likert_response,
    raw_scale,
    PlotLikertError,
)

name = "likertpy"