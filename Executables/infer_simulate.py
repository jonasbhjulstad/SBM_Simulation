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
# from sklearn import metrics
import pandas as pd
matplotlib.use('TkAgg')
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
sys.path.append(Binder_path)
from SIR_SBM import *
Data_dir = Project_root + "/data/SIR_sim/"
Graphs_dir = Data_dir + "Graphs/"

def complete_graph_max_edges(N):
    n_choose_2 = lambda n: n*(n-1)/2
    return int(n_choose_2(N) + N)
def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)
def create_weighted_graph(nodelists, edgelists):
    N_connections = len(edgelists)
    N_communities = len(nodelists)
    N_pop = len(nodelists[0])
    assert complete_graph_max_edges(N_communities) == N_connections

    G = gt.complete_graph(N_communities, directed=False, self_loops=True)
    weights = [len(e) for e in edgelists]
    G.ep['weight'] = G.new_edge_property('int', vals=weights)
    G.vp['vweight'] = G.new_vertex_property('int', vals=[len(n) for n in nodelists])
    #correct weights with self loops
    for i, e in enumerate(G.edges()):
        if e.source() == e.target():
            G.ep['weight'][e] += N_pop
    return G
def create_directed_ccm(N):
    ccm = []
    for i in range(N):
        for j in range(i, N):
            ccm[i*N + j] = (i,j)
    directed_ccm = []
    for i in range(N):
        for j in range(i, N):
            directed_ccm.append((i,j))
            directed_ccm.append((j,i))
    return directed_ccm

def get_p_I_mapping(vcm, N_communities):
    N_connections = complete_graph_max_edges(N_communities)
    directed_ccm = create_directed_ccm(N_communities)
    ecm = ecm_from_vcm(vcm)
    assert len(directed_ccm) == N_connections*2
    p_I_mapping = np.zeros((N_connections*2))
    for i, edge in enumerate(directed_ccm):
        p_I_mapping[i] = vcm[edge[0]]

def index_normalize(indices):
    #get number of unique numbers
    unique = np.unique(indices)
    new_indices = list(range(len(unique)))

    for i, u in enumerate(unique):
        indices[indices == u] = new_indices[i]
    return indices


def structural_compute(N_pop, N_communities, p_in, p_out, seed, N_graphs):
    edgelists, nodelists, ecms_old, vcms_old = generate_N_SBM_graphs(N_pop, N_communities, p_in, p_out, seed, N_graphs)
    N_blocks = []
    entropies = []
    # NMIs = []
    for nlist, elist in zip(nodelists, edgelists):
        G = create_weighted_graph(nlist, elist)
        weight = G.ep['weight']
        state = gt.minimize_blockmodel_dl(G, state_args=dict(recs=[weight], rec_types=["discrete-poisson"]))
        N_blocks.append(state.get_nonempty_B())
        entropies.append(state.entropy())
        vcm = np.array(list(state.get_state()))
        vcm_normalized = index_normalize(vcm)

        a = 1




    return edgelists, nodelists, N_blocks, entropies


def generate_compute(q, N_pop, N_communities, p_in, p_out, seed, idx):
    print("performing inference for idx " + str(idx))
    p = Sim_Param()
    p.N_communities = N_communities
    p.N_pop = N_pop
    p.N_sims = 1
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
    p.N_graphs = 20
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
    edgelists, nlist, N_blocks, entropies = structural_compute(N_pop, N_communities, p_in, p_out, p.seed, p.N_graphs)
    edgelist_flat = [e for elist in edgelists for e in elist]
    print("Running MC-simulations for idx " + str(idx))
    # run(q, p, edgelist_flat, ecm, vcm, N_connections)
    return N_blocks, entropies

if __name__ == '__main__':

    Np = 10
    q = sycl_queue(cpu_selector())

    N_pop = 100
    N_communities = 10
    p_in = 1.0
    p_out = np.linspace(0.07, 0.16, Np)
    p_in = [0.1]*p_out.shape[0]
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

    block_df = pd.DataFrame(np.array(blocklist).T, columns=p_out)
    ent_df = pd.DataFrame(np.array(entropy_list).T, columns=p_out)

    #violin plot
    sns.violinplot(block_df, ax=ax[0], cut=0)
    #limit x to 2 decimals
    ax[0].set_xticklabels([str(round(p, 2)) for p in p_out])
    sns.violinplot(ent_df, ax=ax[1], cut=0, scale='width')
    ax[1].set_xticklabels([str(round(p, 2)) for p in p_out])
    ax[1].set_xlabel("p_out")
    ax[0].set_ylabel("Number of blocks")
    ax[1].set_ylabel("Entropy")
    _ = [x.grid() for x in ax]
    plt.show()
    a = 1
