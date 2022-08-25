#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from aesara import tensor as at
from bokeh.io import output_notebook, show
# from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.TrendComparison import DiffusionTrendComparison
import pysindy as ps
from pydmd import DMD


# In[2]:


def get_trajectory_matrix(trend):
    return np.array([trend[0]['trends']['node_count'][i] for i in range(3)])


# In[3]:


# Network topology
N_pop = 20
g = nx.erdos_renyi_graph(N_pop, .3)
N_sim = 10
# 2° Model selection
models = [ep.SIRModel(g) for i in range(N_sim)]

# 2° Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.01)
cfg.add_model_parameter('gamma', 0.02)
cfg.add_model_parameter("fraction_infected", 0.1)
[model.set_initial_status(cfg) for model in models]
Nt = 200
trends = []
X = np.zeros((N_sim, 3, Nt))
for i, model in enumerate(models):
    iteration = model.iteration_bunch(Nt)
    
    trends.append(model.build_trends(iteration))
    X[i, :, :] = get_trajectory_matrix(trends[-1])
# 2° Simulation execution




# In[4]:


fig, ax = plt.subplots(1)
N_pops = [50, 40]
p_Is = [1.0, .3]
for N, p in zip(N_pops, p_Is):
    illustration_model = nx.erdos_renyi_graph(N, p)
    nx.draw(illustration_model, ax=ax,  width=.1, style='dashed', edge_color='k', node_color='w', edgecolors='k')
    fig.savefig("Erdos_Renyi_Illustration_{pop}_{pER}.svg".format(pop=N, pER=p), format='svg')


# In[5]:


# In[ ]:


x_grouped = [X[:,i,:] for i in range(3)]

# In[ ]:


# In[ ]:


t = [*list(range(Nt))]*N_sim
x_grouped[0].shape


# In[ ]:


X_list = [x for x in X]
# for i in range(3):
#     X_2D[i,:] = np.array([X[j,i,:] for j in range(N_sim)]).T.ravel()
# t = np.hstack([np.repeat(i, N_sim) for i in range(Nt)])


# In[ ]:


reg_model = ps.SINDy(ps.FROLS)

reg_model.fit(X_list, multiple_trajectories=True)


# In[ ]:


sim = reg_model.simulate(x0=X_2D[:,Nt], t=np.linspace(0,200, 10000))
plt.plot(sim)


# In[ ]:




