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
    A class to generate visualizations based on categorical data and associated metrics.

    This class is designed to create Likert-style bar charts using categorical data
    provided in a pandas DataFrame. Each category is represented as a stacked bar,
    with the remaining columns corresponding to numerical metrics.

    Attributes:
        df (pd.DataFrame): The input DataFrame. The DataFrame columns should be like the following example
        | Category | Metric1 | Metric2 | Metric3 |


    Methods:
        likert_plot():
            Generates a horizontal stacked bar chart (Likert chart) to visualize the
            metrics associated with each category.
    """

    def __init__(self, df):
        """
        Init createPlot object

        Args:
            df (DataFrame): Data that will be plot

        Return:
            Plot
        """

        self.df = df
        self.fig = createFigure().createFig()

    def _percentage_df(self):
        # Calcular los porcentajes para cada categoría
        total_counts = self.df.iloc[:, 1:].sum(axis=1)  # Suma total de cada fila
        # print(total_counts)
        self.percentage_df = (
            self.df.iloc[:, 1:].div(total_counts, axis=0) * 100
        )  # Porcentajes
        # Separar categorías negativas (0, 1, 2) y positivas (3, 4, 5)
        self.negative_categories = [0, 1, 2]
        self.positive_categories = [3, 4, 5]
            


    def likert_plot(self):
        """
        Creates and displays a horizontal stacked bar chart (Likert chart).

        This method iterates through each category in the DataFrame's first column
        and adds a horizontal bar for the corresponding metrics in the other columns.
        The bars for all categories are stacked to visualize the distribution of
        metrics across categories.

        Args:
            None
        PlotReturns:
            None: The chart is directly displayed via `fig.show()`.
        """
        self._percentage_df()
        # Inicializar la figura
        for category in self.df["Zoo"].values:
            # Porcentajes negativos
            if self.df["Zoo"][category] in self.negative_categories:
                self.fig.add_trace(
                    go.Bar(
                        y = self.df.columns[1:],
                        x = list(self.percentage_df.loc[self.df["Zoo"] == category].transpose().iloc[:, 0]),# Valores negativos
                        name=f"{category}",  # Nombre del trazo
                        orientation="h",  # Barras horizontales
                    )
                )
            elif self.df["Zoo"][category] in self.positive_categories:  
            # Porcentajes positivos
                self.fig.add_trace(
                        go.Bar(
                            y = self.df.columns[1:],
                            x = list(self.percentage_df.loc[self.df["Zoo"] == category].transpose().iloc[:, 0]),# Valores positivos
                            name=f"{category}",  # Nombre del trazo
                            orientation="h",  # Barras horizontales
                        )
                    )
            else:
                break

        # Configurar diseño del gráfico
        self.fig.update_layout(
            barmode="stack",  # Barras relativas para reflejar Likert
            xaxis_title="Percentage (%)",  # Título del eje x
            yaxis_title="Labels",  # Título del eje y
            # xaxis=dict(
            #     tickformat=".0%", zeroline=True, zerolinecolor="gray"
            # ),  # Centrar en 0%
        )
        self.fig.show()

