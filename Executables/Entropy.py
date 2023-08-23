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

# from Community_Inference import *

fpath = Data_dir + "Graph_0/"

# make fpath directory
if not os.path.exists(fpath):
    os.makedirs(fpath)


def remap(ccm_indices):
    old_indices = []
    new_indices = np.zeros(len(ccm_indices), dtype=int)
    curr_idx = -1
    for i, c in enumerate(ccm_indices):
        if c not in old_indices:
            old_indices.append(c)
        new_indices[i] = np.where(np.array(old_indices) == c)[0][0]
    return new_indices, old_indices

def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def N_graph_inference(pool, N_pop, N_communities, p_in, p_out, seed, Ng):
    #generate seeds
    np.random.seed(seed)

    [edge_lists, node_lists] = generate_N_SBM_graphs_flat(N_pop, N_communities, p_in, p_out, seed, Ng)
    output_dir = Graphs_dir + str(p_out)[:3] + "/"

    #make new directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    [np.savetxt(output_dir + "edge_list_" + str(i) + ".csv", np.array(elist), delimiter=',', fmt="%i") for i, elist in enumerate(edge_lists)]
    [np.savetxt(output_dir + "node_list_" + str(i) + ".csv", np.array(nlist), delimiter=',', fmt="%i") for i, nlist in enumerate(node_lists)]

    Gs = [gt.Graph(directed=False) for _ in range(Ng)]

    [g.add_edge_list(edge_list) for g, edge_list in zip(Gs, edge_lists)]

    states = starmap_with_kwargs(pool, gt.minimize_blockmodel_dl, zip(Gs), repeat(dict(state_args={'deg_corr': False})))
    entropies = [state.entropy() for state in states]
    np.savetxt(output_dir + "entropies.csv", np.array(entropies), delimiter=',')
    ccm_indices = [list(state.get_state()) for state in states]
    remap_indices = [remap(ccm_idx) for ccm_idx in ccm_indices]
    new_indices = [ri[0] for ri in remap_indices]
    old_indices = [ri[1] for ri in remap_indices]
    NMIs = []
    [np.savetxt(output_dir + "ccm_indices_" + str(i) + ".csv", np.array(ccm_idx), delimiter=',', fmt="%i") for i, ccm_idx in enumerate(ccm_indices)]
    [np.savetxt(output_dir + "new_indices_" + str(i) + ".csv", np.array(new_idx), delimiter=',', fmt="%i") for i, new_idx in enumerate(new_indices)]
    [np.savetxt(output_dir + "old_indices_" + str(i) + ".csv", np.array(old_idx), delimiter=',', fmt="%i") for i, old_idx in enumerate(old_indices)]
    [g.save(output_dir + "graph_" + str(i) + ".gt") for i, g in enumerate(Gs)]
    return




if __name__ == '__main__':

    queue = []

    if len(sys.argv) > 1:
        queue = create_sycl_device_queue(sys.argv[1], int(sys.argv[2]))
    else:
        queue = sycl_queue(gpu_selector())

    queues = get_sycl_gpus()
    device_infos = [get_device_info(q) for q in queues]

    _ = [d.print() for d in device_infos]


    N_sim = 2
    N_pop = 100
    N_communities = 5
    p_in = 1.0
    seed = 999
    Nt = 70
    Ng = 20
    p_out = np.arange(0.0, 1.0, 0.1)
    # make directory
    np.random.seed(seed)
    seeds = np.random.randint(0, 100000, 11*11)

    # [edge_lists, node_lists] = generate_N_SBM_graphs_flat(N_pop, N_communities, p_in, p_out, seed, Ng)


    # edge_list = [x for y in edge_lists for x in y]
    # Gs = [gt.Graph(directed=False) for _ in range(Ng)]

    # [g.add_edge_list(edge_list) for g, edge_list in zip(Gs, edge_lists)]

    pool = mp.Pool(mp.cpu_count())
    # write to file
    # G.save(fpath + "/base_graph.gt")

    # plot graph
    # gt.graph_draw(G, output=fpath + "graph.png")
    # states = []
    for po in p_out:
        print("Computing for p_out = " + str(po))
        N_graph_inference(pool, N_pop, N_communities, p_in, po, seed, Ng)
    # states = pool.starmap(gt.minimize_blockmodel_dl, [(g, dict(state_args={'deg_corr': False}, key=True)) for g in Gs])
    # for i, g in enumerate(Gs):
        # states.append(gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=False)))
        # print(i)
    # states = [gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=False)) for g in Gs]


    # state = gt.minimize_blockmodel_dl(G, state_args=dict(deg_corr=False))
    # state.draw(vertex_shape=state.get_blocks(), output=fpath + "state.png")
    # entropies = [state.level_entropy(i) for i in range(len(state.get_levels()))]

    # entropy_tol = 80
    # idx = 0
    # for i, e in enumerate(entropies):
    #     if e < entropy_tol:
    #         idx = i-1
    #         break
    # bmap = state.get_bs()[idx]
    # b_elem_prev = -1

    # # plt.plot(entropies)
    # # plt.show()
    # lv = state.get_levels()
    # state.get_levels()[idx].g.save(fpath + "graph.gt")
    # plot state
    # state.draw(vertex_shape=state.get_blocks(), output=fpath + "state.png")
    # ccm_indices = list(state.get_state())
    # new_idx, old_idx = remap(ccm_indices)

    a = 1
