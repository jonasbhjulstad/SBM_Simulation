import graph_tool.all as gt
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import multiprocessing as mp
from collections import Counter
import json
Project_root = "/home/man/Documents/Sycl_Graph_Old/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/"
sys.path.append(Binder_path)

from SIR_SBM import *
# from libgraph_tool_SIR_SBM import *

# def create_planted(N_pop, N_clusters, p_in, p_out):
#     b = np.concatenate([np.ones(N_pop) for _ in range(N_clusters)])
#     p_SBM = np.zeros(shape=(N_pop, N_pop))
#     for i in range(N_pop):
#         for j in range(N_pop):
#             if b[i] == b[j]:
#                 p_SBM[i, j] = p_in
#             else:
#                 p_SBM[i, j] = p_out
#     return gt.generate_sbm(b, p_SBM, directed=False)
    
fpath = "/home/man/Documents/Sycl_Graph_Old/data/SIR_sim/Graph_0/"
def graph_convert(G):
    G_gt = gt.Graph(directed=True)
    G_gt.add_vertex(len(G.node_list))
    for e in G.edge_list:
        G_gt.add_edge(e._from, e._to)

    return G_gt

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





if __name__ == '__main__':



    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    p_out = np.linspace(0.01, .99, 44)
    N_threads = 4
    seed = 999
    Nt = 70
    Ng = 1
    #make directory
    np.random.seed(seed)
    seeds = np.random.randint(0, 100000, 11*11)



    G = create_planted_SBM(N_pop, N_clusters, p_in, p_out[0], 4, seeds[0])

    gi = graph_convert(G)

    #write to file
    gi.save(fpath + "/base_graph.gt")

    state = gt.minimize_nested_blockmodel_dl(gi)         


    entropies = [state.level_entropy(i) for i in range(len(state.get_levels()))]

    entropy_tol = 80
    idx = 0
    for i, e in enumerate(entropies):
        if e < entropy_tol:
            idx = i-1
            break
    bmap = state.get_bs()[idx]
    b_elem_prev = -1

    # plt.plot(entropies)
    # plt.show()

    state.get_levels()[idx].g.save(fpath + "graph.gt")

    G = remap(G, state, idx)

    p_I0 = 0.1
    p_R = 0.1
    q = sycl_queue(gpu_selector())
    p_I_min = 1e-2
    p_I_max = 1e-1

    p_Is = generate_p_Is(G.N_connections, p_I_min, p_I_max, Nt, seed)
    param = SIR_SBM_Param_t()
    param.p_I = p_Is
    simulate_to_file(G, param, q, fpath, 0, seed)

    json_param = {"p_R": p_R, "p_I0": p_I0, "p_R0": param.p_R0, "p_I_min": p_I_min, "p_I_max": p_I_max, "Nt": Nt, "entropy_tol": entropy_tol, "seed": seed, "N_clusters": N_clusters, "N_pop": N_pop, "p_in": p_in, "p_out": p_out, "hierarchy_idx": idx}
    json_param["entropies"] = entropies

    with open(fpath + "param.json", "w") as f:
        json.dump(json_param, f)
    # levels = state.get_levels()
    # [x.draw(output="a_{}.png".format(i)) for i, x in enumerate(levels)]

    # state.draw(output="a.png")
    # #get entropies
    # a = 1


