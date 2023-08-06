import graph_tool.all as gt
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import multiprocessing as mp
from collections import Counter
import json
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
Data_dir = Project_root + "/data/SIR_sim/"
sys.path.append(Binder_path)

from SIR_SBM import *


def remap(G, state, idx):
    community_map = list(state.project_partition(idx, 0))
    count = Counter(community_map)
    N_count_keys = len(count.keys())
    new_keys = list(range(N_count_keys))
    idx_map = {k: v for k, v in zip(count.keys(), new_keys)}
    #replace indices with linspace(0, N_count_keys, N_count_keys)
    for i, c in enumerate(community_map):
        community_map[i] = int(idx_map[c])

    edge_idx_map = []
    for e in state.get_levels()[idx+1].g.get_edges():
        edge_idx_map.append(Edge_t(idx_map[e[0]], idx_map[e[1]]))

    G.remap(community_map, edge_idx_map)
    return G
