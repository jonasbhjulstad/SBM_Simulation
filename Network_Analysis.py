import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from aesara import tensor as at
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend

if __name__ == '__main__':
    N_pop = 100
    p = .5

    G = nx.erdos_renyi_graph(N_pop, p)
    model = ep.SIRModel(G)

    config = mc.Configuration()
    config.add_model_parameter('beta', 0.1)
    config.add_model_parameter('gamma', 0.1)
    config.add_model_parameter("fraction_infected", 0.1)
    model.set_initial_status(config)

    iterations = [model.iteration_bunch(50) for i in range(1000)]
    trends = [model.build_trends(it) for it in iterations]

    viz = [DiffusionTrend(model, tr) for tr in trends]
    p = viz[0].plot(width=400, height=400)
    show(p)
