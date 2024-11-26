import plotly.graph_objects as go


class createFigure:
    """
    This class is used to create a plot using plotly library.

    Attributes:
        None

    Returns:
        fig (go.Figure) -- Figure object to plot
    """

    def __init__(self):
        pass

    def createFig(self) -> go.Figure:
        fig = go.Figure()
        return fig


class createPlot:
    """


    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(self, df):
        self.df = df

    def likert_plot(self):
        fig = createFigure().createFig()
        for category in self.df["Zoo"].values:
            fig.add_trace(
                go.Bar(
                    y=self.df.columns[1:],
                    x=list(
                        self.df.loc[self.df["Zoo"] == category][
                            list(self.df.columns[1:])
                        ]
                        .transpose()
                        .iloc[:, 0]
                    ),
                    name=str(category),
                    orientation="h",
                )
            )

        fig.update_layout(barmode="stack")
        fig.show()
