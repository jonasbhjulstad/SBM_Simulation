import graph_tool.all as gt
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import multiprocessing as mp
Project_root = "/home/man/Documents/Old_Sycl_Graph/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/"
sys.path.append(Binder_path)
from SIR_SBM import *

def gen_draw_graph(N_pop, N_clusters, p_in, p_out, N_threads, seed):
    G = create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed)

    weights = [len(edgelist) for edgelist in G.edge_lists]


    gt_edgelist = []
    for i, edgelist in enumerate(G.edge_lists):
        id_to = G.connection_targets[i]
        id_from = G.connection_sources[i]
        gt_edgelist.append((id_from, id_to, weights[i]))
    gt_G = gt.Graph(gt_edgelist, eprops=[("weight", "double")])




    #number of edges in gt_G
    a = gt_G.num_edges()
    b = gt_G.num_vertices()
    state = gt.minimize_nested_blockmodel_dl(gt_G, state_args=dict(recs=[gt_G.ep.weight], rec_types=["discrete-binomial"]))
    return gt_G, state

def obtain_graph_entropy(N_pop, N_clusters, p_in, p_out, N_threads, seed):
    G = create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed)

    weights = [len(edgelist) for edgelist in G.edge_lists]


    gt_edgelist = []
    for i, edgelist in enumerate(G.edge_lists):
        id_to = G.connection_targets[i]
        id_from = G.connection_sources[i]
        gt_edgelist.append((id_from, id_to, weights[i]))
    gt_G = gt.Graph(gt_edgelist, eprops=[("weight", "double")])




    #number of edges in gt_G
    a = gt_G.num_edges()
    b = gt_G.num_vertices()
    state = gt.minimize_blockmodel_dl(gt_G, state_args=dict(recs=[gt_G.ep.weight], rec_types=["discrete-binomial"]))
    
    return state.entropy()

def custom_draw(g, state, pi, po):
    state.draw(edge_color=g.ep.weight, ecmap=(matplotlib.cm.inferno, .6),
            eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 2, 8, power=1),
            edge_gradient=[], output=Data_dir + "/Illustrations/graph_" + str(pi) + "_" + str(po) + ".png")
if __name__ == '__main__':

    N_pop = 100
    N_clusters = 10
    p_in = np.linspace(0.0, 1.0, 6)
    p_out = np.linspace(0.0, 1.0, 6)
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
    pool = mp.Pool(processes=8)
    entropies = np.zeros(shape=(len(p_in), len(p_out)))

    for i, pi in enumerate(p_in):
        for j, po in enumerate(p_out):
            print("j: ", j)
            entropies[i, j] = obtain_graph_entropy(N_pop, N_clusters, pi, po, N_threads, np.random.randint(0, 100000))


    #heatmap of entropies
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(entropies, cmap='gray', interpolation='nearest')
    #p_in on x-axis, p_out on y-axis
    ax.set_xticks(np.arange(len(p_in)))
    ax.set_yticks(np.arange(len(p_out)))
    ax.set_xticklabels(p_in)
    ax.set_yticklabels(p_out)
    #set decimal precision to 1
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel("p_in")
    ax.set_ylabel("p_out")
    ax.set_title("Entropy of SBM Graphs")


    #save figure
    fig.savefig(Data_dir + "/Illustrations/entropy_heatmap.png")            



