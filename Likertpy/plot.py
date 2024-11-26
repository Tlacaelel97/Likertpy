import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

# Fake data
df = pd.DataFrame()

names = ["SF Zoo", "LA Zoo"]

d1 = [20, 12]
d2 = [14, 18]
d3 = [23, 29]

df["Zoo"] = names
df["Giraffes"] = d1
df["Orangutans"] = d2
df["Monkeys"] = d3

print(list(df.loc[df["Zoo"] == "SF Zoo"][list(df.columns[1:])]
                .transpose()
                .iloc[:, 0]))
# print(df)
for category in df["Zoo"].values:
    fig.add_trace(
        go.Bar(
            y=df.columns[1:],
            x=list(
                df.loc[df["Zoo"] == category][list(df.columns[1:])]
                .transpose()
                .iloc[:, 0]
            ),
            name=str(category),
            orientation="h"
        )
    )


fig.update_layout(barmode="stack")
fig.show()
