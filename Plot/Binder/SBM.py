#add binder .so to path
SBM_Binder_path = "/home/man/Documents/Old_Sycl_Graph/build/Binders"
Sycl_path = "/opt/intel/oneapi/compiler/2022.2.0/linux/lib"
import matplotlib.pyplot as plt
import sys
from inspect import getmembers, isfunction
import numpy as np
from itertools import product

sys.path.append(SBM_Binder_path)
sys.path.append(Sycl_path)

from SBM_Binder import *
N_pop = [100, 100]
#probabilities


#multiprocessing
from multiprocessing import Pool
import multiprocessing as mp



if __name__ == '__main__':
    N_pop = [100, 100]
    Ng = 10
    N_sims = 500
    G_list = []
    Edge_ID_list = []
    #create all combinations of p_vec and p_vec
    p_vec = np.linspace(0.1,1.0,11)
    p_prod = list(product(p_vec, p_vec))

    N_pop_list = []
    p_SBM_list = []
    for p_pair in p_prod:
        p_SBM_list.append([p_pair[0], p_pair[1], p_pair[1], p_pair[0]])
        N_pop_list.append(N_pop)
    p_I0 = 0.1
    p_R0 = 0.0
    Nt = 100
    
    #p_Is for 1 sim =  Nt*4
    #sample p_Is between p_I_min and p_I_max for N_sims*Ng
    p_Is_min = 1e-4
    p_Is_max = 0.2
    p_R = 0.05
    p_Is_list = np.random.uniform(p_Is_min, p_Is_max, size=(N_sims*Ng, Nt, 4))

    #create Temporal_Param of same size
    tps = [[Temporal_Param(p_Is_list[i, t], p_R) for i in range(N_sims*Ng)] for t in range(Nt)]

    network_data = create_SIR_Bernoulli_SBMs(N_pop_list[:2], p_SBM_list[:2], p_I0, p_R0, False)

    #unpack pairs
    G_list = [network_data[i][0] for i in range(len(network_data))]
    Edge_ID_list = [network_data[i][1] for i in range(len(network_data))]


    #create pool
    pool = Pool(mp.cpu_count())



    
    #list all definitions in SBM
    # p = Temporal_Param()
    
    #convert to np array
    total_traj = np.array(total_traj)
    group_infected = np.array(group_infected)

    fig, ax = plt.subplots(2)
    ax[0].plot(total_traj)
    ax[1].plot(group_infected)
    plt.show()
    # print(traj)
    # fig, ax = plt.subplots(3)
    # for i in range(3):
    #     ax[i].plot(traj[i])
    # plt.show()        


    