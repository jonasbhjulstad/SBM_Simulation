import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.utils import multi_runs
from aesara import tensor as at


# In[2]:


def get_trajectory_matrix(trend):
    return np.array([trend['trends']['node_count'][i] for i in range(3)])


# In[3]:


# Network topology
N_pop = 200
g = nx.erdos_renyi_graph(N_pop, .3)
N_sim = 100
# 2° Model selection
model = ep.SIRModel(g)

# 2° Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.01)
cfg.add_model_parameter('gamma', 0.02)
cfg.add_model_parameter("fraction_infected", 0.1)
model.set_initial_status(cfg)
Nt = 200
X = np.zeros((N_sim, 3, Nt))

#Generate infection sets for the different runs
infection_sets = []
for i in range(N_sim):
    infection_sets.append(model.generate_infection_set(cfg))

trends = multi_runs(model, execution_number=N_sim, iteration_number=Nt, infection_sets=infection_sets, nprocesses=6)    
for i, trend in enumerate(trends):
    X[i, :, :] = get_trajectory_matrix(trend)


np.savetxt('Data/SIR_trajectories.csv', X.reshape(N_sim, -1), delimiter=',')