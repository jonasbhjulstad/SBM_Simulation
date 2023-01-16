import plotly.plotly
from plotly import express as px

fig = px.bar(x=[1, 2, 3], y=[1, 3, 2])
fig.write_html("first_figure.html", auto_open=True)