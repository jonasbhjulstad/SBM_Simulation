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

def gen_draw_graph(N_pop, N_clusters, p_in, p_out, N_threads, seed):
    G = create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed)

    weights = [len(edgelist) for edgelist in G.edge_lists]


    gt_edgelist = []
    for i, edgelist in enumerate(G.edge_lists):
        id_to = G.connection_targets[i]
        id_from = G.connection_sources[i]
        gt_edgelist.append((id_from, id_to, weights[i]))
    # gt_G = gt.Graph(gt_edgelist, eprops=[("weight", "double")])
    v_labels = np.concatenate([np.ones(len(v_list))*i for i, v_list in enumerate(G.vertex_list)])
    gt_G = gt.Graph(np.concatenate(G.edge_lists), directed=False)
    v_prop= gt_G.new_vertex_property("int")
    v_prop.a = v_labels
    
    #number of edges in gt_G
    a = gt_G.num_edges()
    b = gt_G.num_vertices()
    state = gt.minimize_blockmodel_dl(gt_G, state_args=dict(recs=[gt_G.ep.weight], rec_types=["discrete-poisson"]))
    return gt_G, state

def obtain_graph_state(N_pop, N_clusters, p_in, p_out, N_threads, seed):
    G = create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed)

    weights = [len(edgelist) for edgelist in G.edge_lists]


    gt_edgelist = []
    for i, edgelist in enumerate(G.edge_lists):
        id_to = G.connection_targets[i]
        id_from = G.connection_sources[i]
        gt_edgelist.append((id_from, id_to, weights[i]))
    gt_G = gt.Graph(gt_edgelist, eprops=[("weight", "int")])
    # v_labels = np.concatenate([np.ones(len(v_list))*i for i, v_list in enumerate(G.node_list)])
    # gt_G = gt.Graph(np.concatenate([e for e in G.edge_lists if e != []]), directed=False)
    # v_prop= gt_G.new_vertex_property("int")
    # v_prop.a = v_labels
    


    #number of edges in gt_G
    a = gt_G.num_edges()
    b = gt_G.num_vertices()
    state = gt.minimize_nested_blockmodel_dl(gt_G, state_args=dict(eweight=gt_G.ep.weight))
    # state = gt.minimize_blockmodel_dl(gt_G, state_args=dict(recs=[gt_G.ep.weight], rec_types=["discrete-poisson"]))
    return gt_G, state

def custom_draw(g, state, pi, po):
    state.draw(edge_color=g.ep.weight, ecmap=(matplotlib.cm.inferno, .6),
            eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 2, 8, power=1),
            edge_gradient=[], output=Data_dir + "/Illustrations/graph_" + str(pi) + "_" + str(po) + ".png")
if __name__ == '__main__':

    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    p_out = np.linspace(0.0, .99, 8)
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




    for j, po in enumerate(p_out):
        print("j: ", j)
        G, state = obtain_graph_state(N_pop, N_clusters, p_in, po, N_threads, np.random.randint(0, 100000))
        states.append(state)
        entropies.append([s.entropy() for s in state.get_levels()])
        state.draw(output='graph_' + str(p_in) + '_' + str(po) + '.png')
        # weights.append(w)
        # entropy_levels.append([level.entropy() for level in state.get_levels()])
        # N_partitions.append(states[j])
    #heatmap of entropies


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



