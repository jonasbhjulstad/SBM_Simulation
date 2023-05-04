import sys
import numpy as np
import multiprocessing as mp
Project_root = "/home/man/Documents/Sycl_Graph_Old/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/"
sys.path.append(Binder_path)
sys.path.append('/usr/local/lib')
from SIR_SBM import *
from Community_Inference import *
from itertools import combinations, combinations_with_replacement
from math import comb


def generate_planted_graph(N_clusters, N_pop, p_out):
    p_mat = np.zeros((N_clusters, N_clusters))
    #expected number of randomly connected edges between two groups with N_pop members and p_in probability of connection
    # p_in = N_pop*p_in
    cmap = np.concatenate([np.ones(N_pop)*i for i in range(N_clusters)])
    #convert to int
    cmap = cmap.astype(int)
    #max number of edges between vertex groups of N_pop members
    N_out_edges = int(N_pop*(N_pop-1)*p_out)

    #fill matrix probs
    for i in range(N_clusters):
        for j in range(N_clusters):
            if i == j:
                p_mat[i][j] = N_pop*(N_pop-1)
            else:
                p_mat[i][j] = N_pop*N_pop/2


    g = gt.generate_sbm(cmap, p_mat, micro_ers=True, directed=False)
    gt.remove_self_loops(g)

    clabels = g.new_vertex_property("int")
    clabels.a = cmap
    
    # g = gt.generate_sbm(cmap, p_mat, directed=False, self_loops=False)
    return g, clabels



def gt_minimize_wrapper(g, b_init, clabel, N_clusters):
    state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=True, b=b_init, B = N_clusters, clabel=clabel))

    return gt.minimize_blockmodel_dl(state.get_bg(), state_args=dict(deg_corr=True)), state

def generate_compute(N_clusters, N_pop, p_out):
    G, clabel = generate_planted_graph(N_clusters, N_pop, p_out)
    # Gs = pool.starmap(generate_planted_graph, [(N_clusters, N_pop, p_out) for p_out in p_outs])

    state = gt.minimize_blockmodel_dl(G, state_args=dict(deg_corr=True, b=clabel, B = N_clusters, clabel=clabel)).get_block_state()
    c_graph = state.get_bg()

    weights = c_graph.new_edge_property("int")
    weights.a = list(state.eweight)

    community_state = gt.minimize_blockmodel_dl(c_graph, state_args=dict(eweight=weights, deg_corr=False)).get_block_state()

    # states = [gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=False)) for g in Gs]
    # res = pool.starmap(gt_minimize_wrapper, [(g, clabel, clabel) for g, clabel in zip(Gs, clabels)])
    # res = [gt_minimize_wrapper(g, b_init, clabel) for g, clabel in zip(Gs, clabels)]
    # states = [r[1] for r in res]
    # states = [gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=False)) for g in Gs]
    state.draw(output="a.png")
    community_state.draw(output="b.png")
    return community_state

if __name__ == '__main__':

    N_clusters = 10
    N_pop = 100
    p_outs = np.linspace(0,1.0,11)
    pool = mp.Pool(mp.cpu_count())

    # community_states = pool.starmap(generate_compute, [(N_clusters, N_pop, p_out) for p_out in p_outs])
    community_states = [generate_compute(N_clusters, N_pop, p_out) for p_out in reversed(p_outs)]

    N_communities = [s.get_nonempty_B() for s in community_states]



    a = 1