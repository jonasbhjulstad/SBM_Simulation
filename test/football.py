import json
from collections import Counter
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import sys
import graph_tool.all as gt
import os
from itertools import repeat
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
sys.path.append(Binder_path)
from SIR_SBM import *
Data_dir = Project_root + "/data/SIR_sim/"
fpath = Data_dir + "Graph_0/"
# g = gt.collection.data["football"]
p_in = 1.0
p_out = 0.0
N_communities = 10
N_pop = 100
nodes = [[i]*N_pop for i in range(N_communities)]
nodes = [x for y in nodes for x in y]

probs = np.identity(N_communities)*p_in
probs[probs==0] = p_out
probs[0][0] = 0
seed = 1023
# clusters = [gt.complete_graph(N_pop, self_loops = False, directed = True) for _ in N_communities]

# G = gt.Graph(directed=True, self_loops=False)
# for c in clusters:
# #     G = gt.graph_union(G, c)
edgelist, nodelist = generate_planted_SBM_flat(N_pop, N_communities, p_in, p_out, seed)

G = gt.Graph(directed=True)
G.add_edge_list(edgelist)
# gt.remove_self_loops(G)
# G = gt.collection.data["football"]

state_sbm = gt.minimize_blockmodel_dl(G)
nested_state_sbm = gt.minimize_nested_blockmodel_dl(G)

state_sbm.draw(output=fpath + "SBM.png")
nested_state_sbm.draw(output=fpath + "Nested_SBM.png")
a = 1
