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

bipart_max_edges = lambda x: 0.25*x*x
complete_digraph_max_edges = lambda x: x*(x-1)
#generate planted partition
p_in = 1.0
p_out = 0.5
N_communities = 4
N_pop = 10
nodes = [[i]*N_pop for i in range(N_communities)]
nodes = [x for y in nodes for x in y]
prop_in = complete_digraph_max_edges(2*N_pop)
prop_out = bipart_max_edges(2*N_pop)*0.0
probs = np.identity(N_communities)*prop_in
probs[probs==0] = prop_out


#generate graph
seed = 1023
G = gt.generate_sbm(nodes, probs)
gt.remove_self_loops(G)
#draw to file
gt.graph_draw(G, output=Graphs_dir + "SBM.png")
a = 1
