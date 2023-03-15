#add binder .so to path
SBM_Binder_path = "/home/man/Documents/Sycl_Graph_Old/build/Binders/"
Sycl_path = "/opt/intel/oneapi/compiler/2022.2.0/linux/lib"
import matplotlib.pyplot as plt
import sys
from inspect import getmembers, isfunction

sys.path.append(SBM_Binder_path)
sys.path.append(Sycl_path)

from SBM_Binder import *
N_pop = [100, 100]
p_SBM = [0.9, 0.1, 0.1, 0.9]
p_I0 = 0.1
p_R0 = 0.0
Nt=100
if __name__ == '__main__':

    #list all definitions in SBM
    # p = Temporal_Param()
    G, SBM_edge_ids = create_SIR_Bernoulli_SBM(N_pop, p_SBM, p_I0, p_R0, True)
    G.initialize()
    tp = []
    for i in range(Nt):
        t = Temporal_Param()
        t.p_Is = [0.1, 0.1, 0.1]
        t.p_Rs = 0.05
    
    traj = G.simulate(Nt, tp)
    print(traj)
    fig, ax = plt.subplots(3)
    for i in range(3):
        ax[i].plot(traj[i])
    plt.show()        


    