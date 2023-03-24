#add binder .so to path
SBM_Binder_path = "/home/man/Documents/Old_Sycl_Graph/build/Binders"
Sycl_path = "/opt/intel/oneapi/compiler/2022.2.0/linux/lib"
import matplotlib.pyplot as plt
import sys
from inspect import getmembers, isfunction
import numpy as np
from itertools import product
import pandas as pd
figpath = "/home/man/Documents/Sycl_Graph_Old/data/"
sys.path.append(SBM_Binder_path)
sys.path.append(Sycl_path)

from SBM_Binder import *
N_pop = [100, 100]
#probabilities

import glob
import re

#find all csv names in figpath
csv_names = glob.glob(figpath + "*.csv")
#find all csv containing 'traj'
traj_names = [x for x in csv_names if 'traj' in x]
#find all csv containing 'Community'
comm_names = [x for x in csv_names if 'Community' in x]
#find biggest number after SBM_ in traj_names
traj_nums = max([int(re.findall(r'\d+', x)[1]) for x in traj_names]) + 1
#find biggest number before .csv in traj_names
N_networks = max([int(re.findall(r'\d+', x)[0]) for x in comm_names]) + 1

#multiprocessing
from multiprocessing import Pool
import multiprocessing as mp

def simulate(network, tp):
    return network.simulate_groups(tp)


def simulate_parallel(pool, networks, tps):
    [n.initialize() for n in networks]
    return pool.map(simulate, ((n, tp) for n, tp in zip(networks, tps)))

if __name__ == '__main__':

    p_SBM_list = np.genfromtxt(figpath + "p_SBM_list.csv", delimiter=',')
    Nt = 100

    fig0, ax0 = plt.subplots(3, 1, figsize=(10, 10))
    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 10))
    i = 0
    for i in range(N_networks):
        _ = [x.clear() for x in ax0]
        _ = [[x.clear() for x in x2] for x2 in ax1]
        fig0.suptitle("Total states")
        fig1.suptitle("Community infections")
        N_inf = [[], [], [], []]
        for j in range(traj_nums):
            traj = pd.read_csv(figpath + "SBM_" + str(i) + "_traj_" + str(j) + ".csv")
            comm_traj = np.genfromtxt(figpath + "SBM_" + str(i) + "_Community_Infected_" + str(j) + ".csv", delimiter=',')
            ax0[0].plot(traj["S"], color='y')
            ax0[1].plot(traj["I"], color='r')
            ax0[2].plot(traj["R"], color='b')
            for k in range(4):
                #total infections
                N_inf[k].append(np.sum(comm_traj[:,k]))
                ax1[k//2][k%2].plot(comm_traj[:,k], color='r')
        for j in range(4):
            ax1[j//2][j%2].set_title("p = " + str(p_SBM_list[i][j]) + ",avg.inf: " + str(np.mean(N_inf[j])))
        fig0.savefig(figpath + "SBM_" + str(i) + ".png")
        fig1.subplots_adjust(hspace=0.5)
        fig1.savefig(figpath + "SBM_" + str(i) + "_Community.png")
        i += 1