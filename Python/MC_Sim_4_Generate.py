import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.utils import multi_runs
from aesara import tensor as at
from ParameterConfig import DATA_DIR, FIGURE_DIR, ROOT_DIR


# In[2]:


def get_trajectory_matrix(trend):
    return np.array([trend['trends']['node_count'][i] for i in range(3)])

def generate_infection_set(N_pop, p_I0):
    I0 = np.random.choice(N_pop, int(p_I0*N_pop), replace=False)
    return I0

# In[3]:


# Network topology
N_pops = [20, 50, 20, 50]
p_ERs = [.1, .1, 1.0, 1.0]
gs = [nx.erdos_renyi_graph(N_pop, p_ER) for N_pop, p_ER in zip(N_pops, p_ERs)]
N_sim = 1000
# 2° Model selection
models = [ep.SIRModel(g) for g in gs]

# 2° Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.01)
cfg.add_model_parameter('gamma', 0.02)
p_I0 = 0.1
cfg.add_model_parameter("fraction_infected", p_I0)
_ = [model.set_initial_status(cfg) for model in models]
Nt = 200
X = np.zeros((N_sim, 3, Nt))

#Generate infection sets for the different runs
fig, ax = plt.subplots(2,2)
for m, (model, N_pop, p_ER) in enumerate(zip(models, N_pops, p_ERs)):

    infection_sets = []
    for i in range(N_sim):
        infection_sets.append(generate_infection_set(N_pop, p_I0))
    trends = multi_runs(models[m], execution_number=N_sim, iteration_number=Nt, infection_sets=infection_sets, nprocesses=4)    
    for i, trend in enumerate(trends):
        X[i, :, :] = get_trajectory_matrix(trend)


    np.savetxt(DATA_DIR + 'SIR_trajectories_' + str(N_pop) + '_' + str(p_ER) + '.csv', X.reshape(N_sim, -1), delimiter=',')