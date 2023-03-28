import sys
import numpy as np

Project_root = "/home/man/Documents/Old_Sycl_Graph/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/"
sys.path.append(Binder_path)
sys.path.append('/usr/local/lib')
from SIR_SBM import *
if __name__ == '__main__':
    selector = gpu_selector()

    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    p_out = 0.0
    N_threads = 4
    N_sims = 100
    seed = 999
    Nt = 70
    tau = .5
    Ng = 1
    output_dirs = [Data_dir + "Graph_" + str(i) + "/" for i in range(Ng)]
    qs = [sycl_queue(selector) for _ in range(N_sims)]
    #initialize np rng
    np.random.seed(seed)
    #create seeds
    seeds = np.random.randint(0, 1000000, N_sims)

    Gs = create_planted_SBMs(Ng, N_pop, N_clusters, p_in, p_out, N_threads, seed)


    N_community_connections = len(Gs[0].edge_lists)
    p = SIR_SBM_Param()
    p.p_R = 0.1
    p.p_I0 = .1
    p.p_R0 = .0

    p_I_min = 1e-3
    p_I_max = 1e-2

    params = [[p for _ in range(N_sims)] for _ in range(Ng)]

    p_Is = generate_p_Is(N_community_connections, N_sims, Ng, p_I_min, p_I_max, Nt, seed)

    #assign p_I to each param
    for i in range(Ng):
        for j in range(N_sims):
            params[i][j].p_I = p_Is[i][j]
    for i, (p, G) in enumerate(zip(params, Gs)):
        output_dir = Data_dir + "Graph_" + str(i) + "/"
        parallel_simulate_to_file(G, p, qs, output_dir, N_sims, seed)
        alpha, theta_LS, theta_QR = regression_on_datasets(output_dir, N_sims, tau)
        #write alpha, theta_LS, theta_QR to files in output_dir
        np.array(alpha).tofile(output_dir + "alpha.csv", sep=",")
        np.array(theta_LS).tofile(output_dir + "theta_LS.csv", sep=",")
        np.array(theta_QR).tofile(output_dir + "theta_QR.csv", sep=",")


    a = 1