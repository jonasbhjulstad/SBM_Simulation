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
Graphs_dir = Data_dir + "Graphs/"
def remap(ccm_indices):
    old_indices = []
    new_indices = np.zeros(len(ccm_indices), dtype=int)
    curr_idx = -1
    for i, c in enumerate(ccm_indices):
        if c not in old_indices:
            old_indices.append(c)
        new_indices[i] = np.where(np.array(old_indices) == c)[0][0]
    return new_indices, old_indices
bipart_max_edges = lambda x: 0.25*x*x
complete_digraph_max_edges = lambda x: x*(x-1)
#generate planted partition
p_in = 1.0
p_out = np.arange(0.0, 1.0, 0.1)
N_communities = 2
N_pop = 100
nodes = [[i]*N_pop for i in range(N_communities)]
nodes = [x for y in nodes for x in y]
prop_in = complete_digraph_max_edges(2*N_pop)*p_in
prop_out = bipart_max_edges(2*N_pop)*p_out

#generate seeds

seeds = np.random.randint(0, 1000000, len(prop_out))

NMI = []
for prp_out, s in zip(prop_out, seeds):

    probs = np.identity(N_communities)*prop_in

    probs[probs==0] = prp_out
# a = [9]*100
# b = [10]*100

# res = gt.mutual_information(a, b, norm=True)



    G = gt.generate_sbm(nodes, probs)
    #get node labels



    gt.remove_self_loops(G)

    state = gt.minimize_blockmodel_dl(G)

    y = gt.align_partition_labels(list(state.get_state()), nodes)


    NMI.append(gt.mutual_information(y, nodes, norm=True))

plt.plot(p_out, NMI)
plt.show()
a = 1
