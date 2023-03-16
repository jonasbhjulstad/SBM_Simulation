#add binder .so to path
SBM_Binder_path = "/home/man/Documents/Sycl_Graph_Old/build/Binders"
Sycl_path = "/opt/intel/oneapi/compiler/2022.2.0/linux/lib"
import matplotlib.pyplot as plt
import sys
from inspect import getmembers, isfunction
import numpy as np

sys.path.append(SBM_Binder_path)
sys.path.append(Sycl_path)

from SBM_Binder import *
N_pop = [100, 100]
#probabilities



if __name__ == '__main__':
    N_pop = [100, 100]
    Ng = 10
    G_list = []
    Edge_ID_list = []
    p_vec = np.linspace(0.0,1.0,11)
    p_I0 = 0.1
    p_R0 = 0.0
    Nt = 100
    
    tp = []
    for i in range(Nt):
        t = Temporal_Param()
        t.p_Is = [0.1, 0.1, 0.1, 0.1]
        t.p_R = 0.05
        tp.append(t)

    trajectory_list = []
    group_infected_list = []
    for p0 in p_vec:
        for p1 in p_vec:
            p_SBM = [p0, p1, p1, p0]
            Gp = []
            edge_ids = []
            Gp_traj = []
            Gp_inf = []
            for i in range(Ng):
                G, SBM_edge_ids = create_SIR_Bernoulli_SBM(N_pop, p_SBM, p_I0, p_R0, True)
                edge_ids.append(SBM_edge_ids)
                G.initialize()
                traj, g_inf = G.simulate_groups(tp)
                Gp_traj.append(traj)
                Gp_inf.append(g_inf)
            
            trajectory_list.append(Gp_traj)
            group_infected_list.append(Gp_inf)
            Edge_ID_list.append(edge_ids)
    NG_tot = len(G_list)*Ng
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


    