import sys
import numpy as np
import graph_tool.all as gt
Project_root = "/home/man/Documents/Sycl_Graph_Old/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/"
sys.path.append(Binder_path)
sys.path.append('/usr/local/lib')
import os
from SIR_SBM import *
from time import time

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

def obtain_community_inferred_planted_partition(N_pop, N_clusters, p_in, p_out, N_threads, seed):
    gt_G, state, G = obtain_graph_state(N_pop, N_clusters, p_in, p_out, N_threads, seed)
    c_map, N_communities = obtain_p_I_mapping(state, 1)
    G_cmap = rearrange_SBM_with_cmap(c_map, G)
    return G_cmap


def simulate_single_graph(output_dir, N_sims, p_in, p_out, p_I_min, p_I_max, Nt = 56):
    seeds = np.random.randint(0, 1000000, N_sims)

    N_threads = 4
    p = SIR_SBM_Param()
    p.p_R = 0.1
    p.p_I0 = .1
    p.p_R0 = .0

    params = [p for _ in range(N_sims)]


    G_community_mapped = obtain_community_inferred_planted_partition(N_pop, N_clusters, p_in, p_out, N_threads, seed)
    N_connections = len(G_community_mapped.edge_lists)
    N_communities = len(G_community_mapped.node_list)
    p_Is = np.random.uniform(p_I_min, p_I_max, (Nt, N_connections, N_sims))
    for j in range(N_sims):
        params[j].p_I = [p_Is[i, :,j] for i in range(Nt)]

    parallel_simulate_to_file(G_community_mapped, params, qs, output_dir, N_sims, seed)
    alpha, theta_LS, theta_QR = regression_on_datasets(output_dir, N_sims, tau, 0)
    #write alpha, theta_LS, theta_QR to files in output_dir
    np.array(alpha).tofile(output_dir + "alpha.csv", sep=",")
    np.array(theta_LS).tofile(output_dir + "theta_LS.csv", sep=",")
    np.array(theta_QR).tofile(output_dir + "theta_QR.csv", sep=",")
    return N_communities

if __name__ == '__main__':
    selector = gpu_selector()

    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    Ng = 11
    p_out = np.linspace(0,1,11)
    N_threads = 4
    N_sims = 100
    seed = 675
    Nt = 70
    tau = 0.9
    Ng = 10
    p_I_min = 1e-3
    p_I_max = 1e-2
    output_dirs = [Data_dir + "Graph_" + str(i) + "/" for i in range(Ng)]
    qs = [sycl_queue(selector) for _ in range(N_sims)]
    #initialize np rng
    np.random.seed(seed)
    


    for i, po in enumerate(p_out):
        #get 2 decimal po str
        po_str = str(po).split(".")[1]
        #make directory
        po_dir = Data_dir + "/p_out_" + str(i)
        #if not exists
        if not os.path.exists(po_dir):
            os.mkdir(po_dir)

        Graph_communities = []
        for j in range(Ng):
            print("Graph: " + str(j) + " p_out: " + str(po))
            #make directory
            output_dir = Data_dir + "/p_out_" + str(i) + "/Graph_" + str(j) + "/"
            #if not exists
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            #simulate
            #time the simulation
            start = time()
            N_communities = simulate_single_graph(output_dir, N_sims, p_in, po, p_I_min, p_I_max, Nt)
            end = time()
            print("Time: " + str(end - start))
            Graph_communities.append(N_communities)
        #write Graph_communities to file
        np.array(Graph_communities).tofile(po_dir + "/Graph_communities.csv", sep=",")
