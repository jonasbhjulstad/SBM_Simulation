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
# from Community_Inference import *
from SIR_SBM import *
import os

fpath = Data_dir + "Graph_0/"

#make fpath directory
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

if __name__ == '__main__':

    queue = []

    if len(sys.argv) > 1:
        queue = create_sycl_device_queue(sys.argv[1], int(sys.argv[2]))
    else:
        queue = sycl_queue(gpu_selector())

    queues = get_sycl_gpus()
    device_infos = get_device_info(queues)

    _ = [d.print() for d in device_infos]

    N_sim = 20
    N_pop = 100
    N_clusters = 4
    p_in = 1.0
    p_out = 0.5
    N_threads = 4
    seed = 999
    Nt = 70
    Ng = 1
    #make directory
    np.random.seed(seed)
    seeds = np.random.randint(0, 100000, 11*11)



    edge_lists, node_lists = generate_planted_SBM_edges(N_pop, N_clusters, p_in, p_out, seed)
    edge_list = [x for y in edge_lists for x in y]

    G = gt.Graph(directed=False)

    G.add_edge_list(edge_list)

    #write to file
    G.save(fpath + "/base_graph.gt")

    #plot graph
    #gt.graph_draw(G, output=fpath + "graph.png")

    state = gt.minimize_blockmodel_dl(G, state_args=dict(deg_corr=False))
    state.draw(vertex_shape=state.get_blocks(), output=fpath + "state.png")
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
    #plot state
    state.draw(vertex_shape=state.get_blocks(), output=fpath + "state.png")
    ccm_indices = list(state.get_state())
    new_idx, old_idx = remap(ccm_indices)


    Nt = 20

    p = Sim_Param()
    p.N_pop = N_pop
    p.N_clusters = N_clusters
    p.p_in = p_in
    p.p_out = p_out
    p.Nt = Nt
    p.p_R0 = 0.0
    p.p_I0 = 0.1
    p.sim_idx = 0
    p.seed = seed
    p_I_min = 1e-6
    p_I_max = 1e-2
    p.p_R = 0.1
    p.max_infection_samples = 1000

    parallel_excite_simulate(p, new_idx, edge_list, p_I_min, p_I_max, fpath, N_sim, queue, True)
    a = 1
