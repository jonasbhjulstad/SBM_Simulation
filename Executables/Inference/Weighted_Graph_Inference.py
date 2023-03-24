#add binder .so to path
SBM_Binder_path = "/home/man/Documents/Old_Sycl_Graph/build/Binders"
Sycl_path = "/opt/intel/oneapi/compiler/2022.2.0/linux/lib"
import matplotlib.pyplot as plt
import sys
from inspect import getmembers, isfunction
import numpy as np
from itertools import product

from graph_tool.all import *

figpath = "/home/man/Documents/Sycl_Graph_Old/data/"
sys.path.append(SBM_Binder_path)
sys.path.append(Sycl_path)

from SBM_Binder import *
#probabilities


#multiprocessing
from multiprocessing import Pool
import multiprocessing as mp

def simulate(network, tp):
    return network.simulate_groups(tp)


def simulate_parallel(pool, networks, tps):
    [n.initialize() for n in networks]
    return pool.map(simulate, ((n, tp) for n, tp in zip(networks, tps)))

if __name__ == '__main__':
    N_pop = 20
    p_in = 1.0
    N_sims = 3
    #create all combinations of p_vec and p_vec
    # p_out = np.arange(0.1,1.0,0.1)
    # p_out = np.logspace(-4,1,5)
    p_out = [0.1, 0.5, 1.0]

    N_clusters = 100

    Gs = [generate_planted_partition(N_pop, N_clusters,  p_in, po, False) for po in p_out]

    #get the sum of edges in each sublist
    g_edges = [np.reshape([len(g_edges) for g_edges in G], (N_clusters, N_clusters)) for G in Gs]
    g_edges_nz = [np.nonzero(g_edges) for g_edges in g_edges]



    gs = [Graph(np.array([gnz[0], gnz[1], ge[gnz]]).T, eprops=[("weight", "double")]) for ge, gnz in zip(g_edges, g_edges_nz)]
    states = []
    for i, g in enumerate(gs):

        state_arg = dict(eweight=g.edge_properties["weight"])

        state = minimize_nested_blockmodel_dl(g, state_args=state_arg)
        state.draw(output="planted_" + str(i) + ".pdf")
        states.append(state)

    
    #print summaries
    for state in states:
        state.print_summary()
        print("entropy: ", state.entropy())


    a = 1


    