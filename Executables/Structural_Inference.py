import graph_tool.all as gt
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import multiprocessing as mp
Project_root = "/home/man/Documents/Sycl_Graph_Old/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/"
sys.path.append(Binder_path)
from SIR_SBM import *


def obtain_graph_state(N_pop, N_clusters, p_in, p_out, N_threads, seed):
    G = create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed)

    weights = [len(edgelist) for edgelist in G.edge_lists]


    gt_edgelist = []
    for i, edgelist in enumerate(G.edge_lists):
        id_to = G.connection_targets[i]
        id_from = G.connection_sources[i]
        gt_edgelist.append((id_from, id_to, weights[i]))
    gt_G = gt.Graph(gt_edgelist, eprops=[("weight", "int")])

    


    #number of edges in gt_G
    a = gt_G.num_edges()
    b = gt_G.num_vertices()
    state = gt.minimize_nested_blockmodel_dl(gt_G, state_args=dict(eweight=gt_G.ep.weight))
    # state = gt.minimize_blockmodel_dl(gt_G, state_args=dict(recs=[gt_G.ep.weight], rec_types=["discrete-poisson"]))
    return gt_G, state, G

def obtain_p_I_mapping(state, level):
    l_state = state.get_levels()[level]
    blocks = list(l_state.get_blocks())
    block_idx = np.unique(blocks)
    N_communities = len(block_idx)
    desired_idx = list(range(N_communities))
    
    res_idx = []

    for i in range(len(blocks)):
        w = np.where(block_idx == blocks[i])[0][0]
        res_idx.append(desired_idx[w])
    

    return res_idx, N_communities

def obtain_community_inferred_planted_partition(N_pop, N_clusters, p_in, p_out, N_threads, seed)
    gt_G, state, G = obtain_graph_state(N_pop, N_clusters, p_in, p_out, N_threads, seed)
    c_map, N_communities = obtain_p_I_mapping(state, 1)
    G_cmap = rearrange_SBM_with_cmap(c_map, G)
    return G_cmap

if __name__ == '__main__':

    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    p_out = np.linspace(0.0, .99, 2)
    N_threads = 4
    seed = 999
    Nt = 70
    tau = .5
    Ng = 1
    #make directory
    np.random.seed(seed)
    seeds = np.random.randint(0, 100000, 11*11)

    

    # Gs = []
    # states = []
    # #call gen_draw_graph for each p_in, p_out

    # entropies = np.zeros(shape=(len(p_in), len(p_out)))
    # for i, pi in enumerate(p_in):
    #     for j, po in enumerate(p_out):
    #         G, state = gen_draw_graph(N_pop, N_clusters, pi, po, N_threads, seed)
    #         # custom_draw(G, state, pi, po)
    #         Gs.append(G)
    #         states.append(state)
    #         entropies[i, j] = state.entropy()

    
    #create pool
    entropies = []
    #number of partitions
    N_partitions = []
    pool = mp.Pool(processes=8)
    states = []
    weights = []
    entropy_levels = []



    

    c_map, N_communities = obtain_p_I_mapping(state, 1)

    G_cmap = rearrange_SBM_with_cmap(c_map, G)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, e in enumerate(entropies):
        #increasing black color
        plt.plot(e, color=(0, 0, 0, i/len(entropies)))
    
    plt.show()
    #p_in on x-axis, p_out on y-axis
    # plt.plot(p_out, entropies)
    # #set decimal precision to 1
    # ax.tick_params(axis='both', which='major', labelsize=8)
    # ax.set_xlabel("p_in")
    # ax.set_ylabel("p_out")
    # ax.set_title("Entropy of SBM Graphs")


    # #save figure
    # fig.savefig(Data_dir + "/Illustrations/entropy_heatmap.png")            



