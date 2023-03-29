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

def create_planted(N_pop, N_clusters, p_in, p_out):
    b = np.concatenate([np.ones(N_pop) for _ in range(N_clusters)])
    p_SBM = np.zeros(shape=(N_pop, N_pop))
    for i in range(N_pop):
        for j in range(N_pop):
            if b[i] == b[j]:
                p_SBM[i, j] = p_in
            else:
                p_SBM[i, j] = p_out
    return gt.generate_sbm(b, p_SBM, directed=False)
    


if __name__ == '__main__':



    N_pop = 100
    N_clusters = 10
    p_in = 1.0
    p_out = np.linspace(0.01, .99, 44)
    N_threads = 4
    seed = 999
    Nt = 70
    tau = .5
    Ng = 1
    #make directory
    np.random.seed(seed)
    seeds = np.random.randint(0, 100000, 11*11)

    G = create_planted(N_pop, N_clusters, p_in, p_out[0])

    state = gt.minimize_nested_blockmodel_dl(G, deg_corr=False, B_min=2, B_max=2, mcmc_args=dict(niter=1000))         

    state.draw(output='a.png')


