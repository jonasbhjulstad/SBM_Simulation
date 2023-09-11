import json
from collections import Counter
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import sys
import graph_tool.all as gt
import os
import seaborn as sns
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
def structural_compute(N_pop, N_communities, p_in, p_out, seed, idx):
    edgelists, nodelists, ecm, vcm = generate_planted_SBM_edges(N_pop, N_communities, p_in, p_out, seed)
    G = create_weighted_graph(edgelists, nodelists)
    weight = G.ep['weight']
    state = gt.minimize_blockmodel_dl(G, state_args=dict(recs=[weight], rec_types=["discrete-poisson"]))
    N_blocks = state.get_nonempty_B()
    entropies = state.entropy()
    return edgelists, nodelists, ecm, vcm, N_blocks, entropies

def complete_graph_max_edges(N):
    return int(n_choose_2(N))

def generate_compute(q, N_pop, N_communities, p_in, p_out, seed, idx):
    print("performing inference for idx " + str(idx))
    p = Sim_Param()
    p.N_communities = N_communities
    p.N_pop = N_pop
    p.N_sims = 2
    p.p_in = p_in
    p.p_out = p_out
    p.Nt = 5
    p.Nt_alloc = 5
    p.p_R0 = 0.0
    p.p_R = 0.1
    p.p_I0 = 0.1
    p.p_I_min = .0001
    p.p_I_max = .1
    p.seed = seed
    p.N_graphs = 2
    p.output_dir = Data_dir + "p_out_" + str(p_out)[:4] + "/"
    p.compute_range=sycl_range_1(p.N_graphs*p.N_sims)
    p.wg_range=sycl_range_1(p.N_graphs*p.N_sims)
    seeds = np.random.randint(0, 100000, p.N_graphs)
    #generate graphs
    edgelists = []
    ecms = []
    vcms = []
    entrops = []
    N_blocks = []
    N_connections = complete_graph_max_edges(N_communities)
    for i in range(p.N_graphs):
        elist, nlist, ecm, vcm, Nb, entrop = structural_compute(N_pop, N_communities, p_in, p_out, seeds[i], i)
        edgelists.append([x for y in elist for x in y])
        ecms.append(ecm)
        vcms.append(vcm)
        entrops.append(entrop)
        N_blocks.append(Nb)

    print("Running MC-simulations for idx " + str(idx))
    run(q, p, edgelists, ecms, vcms, N_connections)
    return N_blocks, entropies

if __name__ == '__main__':

    Np = 10
    q = sycl_queue(cpu_selector())

    N_pop = 100
    N_communities = 2
    p_in = 1.0
    p_out = np.linspace(0.0, 0.1, Np)
    p_in = np.max(p_out)-p_out
    Gs = []
    states = []
    N_blocks = []
    entropies = []
    fig, ax = plt.subplots(2)
    seeds = np.random.randint(0, 100000, Np)
    pool = mp.Pool(int(mp.cpu_count()/2))
    # for i in range(N_samples):
    # res = pool.starmap(generate_compute, zip(qs, [N_pop]*Np, [N_communities]*Np, p_in, p_out, seeds, range(Np)))

    blocklist = []
    entropy_list =[]
    for i in range(Np):
        blocks, entrops = generate_compute(q, N_pop, N_communities, p_in[i], p_out[i], seeds[i], i)
        blocklist.append(blocks)
        entropy_list.append(entrops)
    # [ax[0].plot(p_out, nb) for nb in N_blocks]
    # [ax[1].plot(p_out, e) for e in entropies]
    # state.draw(output=Data_dir + "state.png")

    # plt.show()

    #violin plot
    sns.violinplot(data=blocklist, ax=ax[0])
    sns.violinplot(data=entropy_list, ax=ax[1])
    plt.show()
    a = 1
