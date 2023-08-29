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
def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)
def n_choose_2(n):
    return n*(n-1)/2
def create_weighted_graph(edgelists, nodelists):
    N_connections = len(edgelists)
    N_communities = len(nodelists)
    N_pop = len(nodelists[0])
    assert (n_choose_2(N_communities) + N_communities) == N_connections

    G = gt.complete_graph(N_communities, directed=False, self_loops=True)
    weights = [len(e) for e in edgelists]
    G.ep['weight'] = G.new_edge_property('int', vals=weights)
    G.vp['vweight'] = G.new_vertex_property('int', vals=[len(n) for n in nodelists])
    #correct weights with self loops
    for i, e in enumerate(G.edges()):
        if e.source() == e.target():
            G.ep['weight'][e] += N_pop
    return G

def generate_compute(N_pop, N_communities, p_in, p_out, seed):
    edgelists, nodelists = generate_planted_SBM_edges(N_pop, N_communities, p_in, p_out, seed)
    G = create_weighted_graph(edgelists, nodelists)
    weight = G.ep['weight']
    state = gt.minimize_blockmodel_dl(G, state_args=dict(recs=[weight], rec_types=["discrete-poisson"]))
    N_blocks = state.get_nonempty_B()
    entropies = state.entropy()
    return N_blocks, entropies

if __name__ == '__main__':

    N_sim = 2
    N_pop = 20
    N_communities = 100
    p_in = 1.0
    seed = 999
    Nt = 70
    Ng = 2
    p_out = np.linspace(0.0, 1.0, 21)
    p_in = 1 - p_out
    Gs = []
    states = []
    N_blocks = []
    entropies = []
    fig, ax = plt.subplots(2)
    seeds = np.random.randint(0, 100000, len(p_out))
    pool = mp.Pool(int(mp.cpu_count()/2))
    N_samples = 3
    entropies = np.zeros((N_samples, len(p_out)))
    N_blocks = np.zeros((N_samples, len(p_out)))
    for i in range(N_samples):
        print("Computing for sample " + str(i))
        res = pool.starmap(generate_compute, zip(repeat(N_pop), repeat(N_communities), p_in, p_out, seeds))
        entropies[i,:] = [r[1] for r in res]
        N_blocks[i,:] = [r[0] for r in res]

    [ax[0].plot(p_out, nb) for nb in N_blocks]
    [ax[1].plot(p_out, e) for e in entropies]
    # state.draw(output=Data_dir + "state.png")

    plt.show()
    a = 1
