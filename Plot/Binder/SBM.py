#add binder .so to path
SBM_Binder_path = "/home/man/Documents/Old_Sycl_Graph/build/Binders"
Sycl_path = "/opt/intel/oneapi/compiler/2022.2.0/linux/lib"
import matplotlib.pyplot as plt
import sys
from inspect import getmembers, isfunction
import numpy as np
from itertools import product

figpath = "/home/man/Documents/Old_Sycl_Graph/data/"
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
    N_pop = 100
    N_clusters = 10
    N_sims = 20
    #p_Is for 1 sim =  Nt*4
    #sample p_Is between p_I_min and p_I_max for N_sims*Ng
    p_Is_min = 1e-4
    p_Is_max = 0.2
    p_R = 0.05
    p_out = np.logspace(-2,0,4)
    Ng = len(p_out)
    Nt = 70
    p_in = 1.0
    p_I0 = 0.1
    p_Is_list = np.random.uniform(p_Is_min, p_Is_max, size=(Ng, N_sims, Nt, 4))

    #create Temporal_Param of same size
    tps = [[[Temporal_Param(p_Is_list[ng,ns, t], p_R) for t in range(Nt)] for ng in range(Ng)] for ns in range(N_sims)]

    networks = [create_SIR_Bernoulli_planted(N_pop, N_clusters, p_in, po, p_I0, 0.0, False) for po in p_out]
        

    # networks = create_SIR_Bernoulli_SBMs(N_pop_list, p_SBM_list, p_I0, p_R0, False)

    #total byte size of networks:
    print("Total byte size of networks: ", np.sum([n.byte_size() for n in networks]))

    #unpack pairs
    Edge_ID_list = [n.SBM_ids for n in networks]

    simulation_data = simulate_N_parallel_to_file(networks, tps, figpath)

    
    #list all definitions in SBM
    # p = Temporal_Param()

    



    # print(traj)
    # fig, ax = plt.subplots(3)
    # for i in range(3):
    #     ax[i].plot(traj[i])
    # plt.show()        


    