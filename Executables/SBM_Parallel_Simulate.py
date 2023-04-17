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
fpath = Data_dir + "Graph_0/"

def get_edgelists(g, cmap, N_clusters):
    #iterate over edges in g
    #create dict map for all combinations of communities
    community_idx = list(range(N_clusters))
    #all combinations of communities with self-loops
    combs = list(combinations_with_replacement(community_idx, 2))
    # combs = list(combinations(community_idx, 2))
    #create dict map
    connection_map = {}
    for c in combs:
        connection_map[c] = []
    

    ecm = [(int(cmap[e[0]]), int(cmap[e[1]])) for e in g.iter_edges()]
    edges = g.get_edges()

    for ecm_elem, e in zip(ecm, edges):
        #if ecm_elem is key in connection_map:
        if ecm_elem in connection_map:
            connection_map[ecm_elem].append(Edge_t(e[0], e[1]))
        else:
            connection_map[(ecm_elem[1], ecm_elem[0])].append(Edge_t(e[0], e[1]))
    #sort edges by ecm
    #convert dict to list
    connection_map = [connection_map[c] for c in combs]
    return connection_map


def generate_planted_graph(N_clusters, N_pop, p_out):
    p_mat = np.zeros((N_clusters, N_clusters))
    #expected number of randomly connected edges between two groups with N_pop members and p_in probability of connection
    # p_in = N_pop*p_in
    cmap = np.concatenate([np.ones(N_pop)*i for i in range(N_clusters)])

    ks = [gt.complete_graph(N_pop, directed=False, self_loops=False) for _ in range(N_clusters)]
    #remove self loops
    for i in range(len(ks)):
        gt.remove_self_loops(ks[i])
    N_k_edges = ks[0].num_edges()
    g = gt.Graph(directed=False)
    for k in ks:
        g = gt.graph_union(g, k)
    #binomial sample number of edges
    N_pop_tot = N_pop*N_clusters
    
    N_out_edges = np.random.binomial(N_pop_tot*(N_pop_tot-1)/2, p_out) - N_clusters*N_k_edges
    gt.add_random_edges(g, N_out_edges, self_loops=False)
    
    # g = gt.generate_sbm(cmap, p_mat, directed=False, self_loops=False)
    edgelists = get_edgelists(g, cmap, N_clusters)
    nodes = g.get_vertices()
    nodelists = np.array_split(nodes, N_clusters)
    return nodelists, edgelists, g



def gt_to_sycl_graph(g, N_clusters):
    #get vertices in lists grouped by labels
    v = g.get_vertices()
    #split into N_clusters lists
    v_split = np.array_split(v, N_clusters)


if __name__ == '__main__':
    selector = gpu_selector()

    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    p_outs = np.linspace(0.6,0.7,5)
    N_threads = 4
    N_sims = 2
    seed = 675
    Nt = 56
    tau = .95
    Ng = 1
    output_dirs = [Data_dir + "Graph_" + str(i) + "/" for i in range(Ng)]
    qs = [sycl_queue(selector) for _ in range(N_sims)]
    #initialize np rng
    np.random.seed(seed)
    #create seeds
    seeds = np.random.randint(0, 1000000, N_sims)
    output_dir = Data_dir + "Graph_" + str(0) + "/"
    pool = mp.Pool(processes=N_threads)

    # Gs = create_planted_SBMs(Ng, N_pop, N_clusters, p_in, p_out, N_threads, seed)
    Gs = []
    gis = []

    # res = pool.starmap(generate_planted_graph, [(N_clusters, N_pop, p_out) for p_out in p_outs])

    res = [generate_planted_graph(N_clusters, N_pop, p_out) for p_out in p_outs]

    for r in res:
        Gs.append(SBM_Graph_t(r[0], r[1]))
        gis.append(r[2])
    # gis = pool.starmap(generate_planted_graph, [(N_clusters, N_pop, p_in, p_out) for _ in range(Ng)])
    

    # Gs = [Sycl_Graph_t(gi, tau) for gi in gis]

    #write to file
    _ = [gi.save(odir + "/base_graph.gt") for gi, odir in zip(gis, output_dirs)]

    #call gt.minimize nested blockmodel dl on each graph
    Bs = [gt.BlockState(G, deg_corr=False) for G in gis] 

    def gt_minimize_wrapper(g, b):
        gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=False))
        return b

    states = pool.map(gt.minimize_blockmodel_dl, gis)

    # def gt_minimize_wrapper(g, b):
    #     gt.minimize_nested_blockmodel_dl(g, blockstate=b)
    #     return b
    # states = pool.map(gt.minimize_nested_blockmodel_dl, (gis, Bs))

    # _ = [state.draw(output_dir="a_{}.png".format(i)) for i, state in enumerate(states)]
    # level_entropies = [[state.level_entropy(i) for i in range(len(state.get_levels()))] for state in states]
    
    # entropy_tol = 80
    
    # entropy_tol_indices = []
    # for entropies, state in zip(level_entropies, states):
    #     idx = 0
    #     for i, e in enumerate(entropies):
    #         if e < entropy_tol:
    #             idx = i-1
    #             break
    #     entropy_tol_indices.append(idx)

    # plt.plot(entropies)
    # plt.show()

    [state.get_bg().save(odir + "graph.gt") for state, odir in zip(states, output_dirs)]

    # for i in range(len(Gs)):
    #     Gs[i] = remap(Gs[i], states[i], entropy_tol_indices[i])
    
    #array to file
    _ = [G.ccmap_write(odir + "ccmap.csv") for G, odir in zip(Gs, output_dirs)]
    p_R = 0.1




    N_community_connections = Gs[0].N_connections
    p = SIR_SBM_Param_t()
    p.p_R = 0.1
    p.p_I0 = .1
    p.p_R0 = .0

    p_I_min = 1e-3
    p_I_max = 1e-2

    params = [[p for _ in range(N_sims)] for _ in range(Ng)]
    json_param = {"p_I_min": p_I_min, "p_I_max": p_I_max, "Nt": Nt, "seed": seed, "N_clusters": N_clusters, "N_pop": N_pop, "p_in": p_in, "p_out": p_out}


    p_Is = [generate_p_Is(G.N_connections, N_sims, p_I_min, p_I_max, Nt, seed) for G in Gs]

    #assign p_I to each param
    for i in range(Ng):
        for j in range(N_sims):
            params[i][j].p_I = p_Is[i][j]
    output_dirs = []
    for i, (p, G) in enumerate(zip(params, Gs)):
        output_dirs.append(Data_dir + "Graph_" + str(i) + "/")
        parallel_simulate_to_file(G, p, qs, output_dirs[i], N_sims, seed)
        theta_LS, theta_QR = regression_on_datasets(output_dirs[i], N_sims, tau, 0)
        #write alpha, theta_LS, theta_QR to files in output_dir
        # np.array(alpha).tofile(output_dir + "alpha.csv", sep=",")
        np.array(theta_LS).tofile(output_dir + "theta_LS.csv", sep=",")
        np.array(theta_QR).tofile(output_dir + "theta_QR.csv", sep=",")


    with open(fpath + "param.json", "w") as f:
        json.dump(json_param, f)


    a = 1