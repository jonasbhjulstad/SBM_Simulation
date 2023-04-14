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
fpath = Data_dir + "Graph_0/"
if __name__ == '__main__':
    selector = gpu_selector()

    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    p_out = 1.0
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

    Gs = create_planted_SBMs(Ng, N_pop, N_clusters, p_in, p_out, N_threads, seed)

    N_community_connections = Gs[0].N_connections
    p = SIR_SBM_Param_t()
    p.p_R = 0.1
    p.p_I0 = .1
    p.p_R0 = .0

    p_I_min = 1e-3
    p_I_max = 1e-2

    params = [[p for _ in range(N_sims)] for _ in range(Ng)]

    # p_Is = generate_p_Is(N_community_connections, N_sims, Ng, p_I_min, p_I_max, Nt, seed)

    # #assign p_I to each param
    # for i in range(Ng):
    #     for j in range(N_sims):
    #         params[i][j].p_I = p_Is[i][j]
    # output_dirs = []
    # for i, (p, G) in enumerate(zip(params, Gs)):
    #     output_dirs.append(Data_dir + "Graph_" + str(i) + "/")
    #     parallel_simulate_to_file(G, p, qs, output_dirs[i], N_sims, seed)
    #     theta_LS, theta_QR = regression_on_datasets(output_dirs[i], N_sims, tau, 0)
    #     #write alpha, theta_LS, theta_QR to files in output_dir
    #     # np.array(alpha).tofile(output_dir + "alpha.csv", sep=",")
    #     np.array(theta_LS).tofile(output_dir + "theta_LS.csv", sep=",")
    #     np.array(theta_QR).tofile(output_dir + "theta_QR.csv", sep=",")


    

    gis = [graph_convert(G) for G in Gs]

    #write to file
    _ = [gi.save(odir + "/base_graph.gt") for gi, odir in zip(gis, output_dirs)]

    pool = mp.Pool(processes=N_threads)
    #call gt.minimize nested blockmodel dl on each graph
    states = pool.map(gt.minimize_nested_blockmodel_dl, gis)


    level_entropies = [[state.level_entropy(i) for i in range(len(state.get_levels()))] for state in states]

    entropy_tol = 80
    
    entropy_tol_indices = []
    for entropies, state in zip(level_entropies, states):
        idx = 0
        for i, e in enumerate(entropies):
            if e < entropy_tol:
                idx = i-1
                break
        entropy_tol_indices.append(idx)

    # plt.plot(entropies)
    # plt.show()

    [state.get_levels()[idx].g.save(odir + "graph.gt") for state, odir in zip(states, output_dirs)]

    for i in range(len(Gs)):
        Gs[i] = remap(Gs[i], states[i], entropy_tol_indices[i])
    
    #array to file
    _ = [G.ccmap_write(odir + "ccmap.csv") for G, odir in zip(Gs, output_dirs)]
    p_R = 0.1
    json_param = {"p_I_min": p_I_min, "p_I_max": p_I_max, "Nt": Nt, "entropy_tol": entropy_tol, "seed": seed, "N_clusters": N_clusters, "N_pop": N_pop, "p_in": p_in, "p_out": p_out, "hierarchy_idx": idx}
    json_param["entropies"] = entropies

    with open(fpath + "param.json", "w") as f:
        json.dump(json_param, f)


    

    a = 1