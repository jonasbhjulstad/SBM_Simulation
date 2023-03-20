#add binder .so to path
SBM_Binder_path = "/home/man/Documents/Sycl_Graph_Old/build/Binders"
Sycl_path = "/opt/intel/oneapi/compiler/2022.2.0/linux/lib"
import matplotlib.pyplot as plt
import sys
from inspect import getmembers, isfunction
import numpy as np
from itertools import product

figpath = "/home/man/Documents/Sycl_Graph_Old/data/"
sys.path.append(SBM_Binder_path)
sys.path.append(Sycl_path)

from SBM_Binder import *
N_pop = [100, 100]
#probabilities


#multiprocessing
from multiprocessing import Pool
import multiprocessing as mp

def simulate(network, tp):
    return network.simulate_groups(tp)


def simulate_parallel(pool, networks, tps):
    [n.initialize() for n in networks]
    return pool.map(simulate, ((n, tp) for n, tp in zip(networks, tps)))

if __name__ == '__main__':
    N_pop = [100, 100]
    N_sims = 3
    #create all combinations of p_vec and p_vec
    p_vec = np.linspace(0.1,1.0,5)
    p_prod = list(product(p_vec, p_vec))

    N_pop_list = []
    p_SBM_list = []
    for p_pair in p_prod:
        p_SBM_list.append([p_pair[0], p_pair[1], p_pair[1], p_pair[0]])
        N_pop_list.append(N_pop)
    p_I0 = 0.1
    p_R0 = 0.0
    Nt = 100
    Ng = len(p_prod)
    
    #p_SBM_list to file
    p_SBM_list = np.array(p_SBM_list)
    np.savetxt(figpath + "p_SBM_list.csv", p_SBM_list, delimiter=",")

    #p_Is for 1 sim =  Nt*4
    #sample p_Is between p_I_min and p_I_max for N_sims*Ng
    p_Is_min = 1e-4
    p_Is_max = 0.2
    p_R = 0.05
    p_Is_list = np.random.uniform(p_Is_min, p_Is_max, size=(Ng, N_sims, Nt, 4))

    #create Temporal_Param of same size
    tps = [[[Temporal_Param(p_Is_list[ng,ns, t], p_R) for t in range(Nt)] for ng in range(Ng)] for ns in range(N_sims)]

    networks = create_SIR_Bernoulli_SBMs(N_pop_list, p_SBM_list, p_I0, p_R0, False)

    #total byte size of networks:
    print("Total byte size of networks: ", sum([networks[i].byte_size() for i in range(len(networks))]))

    #unpack pairs
    Edge_ID_list = [networks[i].SBM_ids for i in range(len(networks))]


    simulation_data = simulate_N_parallel_to_file(networks, tps, figpath)

    
    #list all definitions in SBM
    # p = Temporal_Param()

    



    # print(traj)
    # fig, ax = plt.subplots(3)
    # for i in range(3):
    #     ax[i].plot(traj[i])
    # plt.show()        


    