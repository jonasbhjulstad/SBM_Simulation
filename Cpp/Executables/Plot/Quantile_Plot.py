# Plot ../data/traj_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys

# BINDER_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/build/Binders/"
# sys.path.insert(0, BINDER_DIR)
# from pyFROLS import *

if __name__ == '__main__':  
    arglist = str(sys.argv)
    if (len(sys.argv) > 1):
        network_type = "SIS"
        statenames = ["S", "I"]
    else:
        network_type = "SIR"
        statenames = ["S", "I", "R"]
    N_pop = 1000
        # read and plot all csv in ../data
        # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    cwd = os.path.dirname(os.path.realpath(__file__))

    DATA_DIR = cwd + '/../../data/'
    FIGURE_DIR = cwd + '/../../figures/'

    # find all csv in data_path
    files = glob.glob(DATA_DIR + "Bernoulli_" + network_type + "_MC_" + str(N_pop) + "_1_*.csv")
    qr_files = glob.glob(DATA_DIR + "Quantile_Simulation_" + network_type + "*.csv")
    er_files = glob.glob(DATA_DIR + "ERR_Simulation_" + network_type + "*.csv")
    # sort q_files according to float in name
    fig, ax = plt.subplots(len(statenames) + 1)
    N_files = 100
    dfs = [pd.read_csv(f, delimiter=",") for f in files[:N_files]]

    qr_dfs = [pd.read_csv(qrf) for qrf in qr_files[:N_files]]
    er_dfs = [pd.read_csv(erf) for erf in er_files[:N_files]]
    
    for j, df in enumerate(dfs):
        for i, name in enumerate(statenames):
            ax[i].plot(df["t"], df[name], color='gray', alpha=.2)
        if (j < 10):
            ax[-1].plot(df["t"][:-1], df["p_I"][:-1], color='gray', alpha=.2)
        if (np.any(df["t"] > 101)):
            a = 1

    # er_dfs = qr_dfs
    fig1, ax1 = plt.subplots(4)
    for (qr, er) in zip(er_dfs, er_dfs):
        for i, name in enumerate(statenames):
            ax1[i].plot(qr["t"][:-1], qr[name][:-1], color='k', alpha=.3)
            ax1[i].plot(er["t"][:-1], er[name][:-1], color='r', alpha=.3)
        ax1[-1].plot(qr["t"][:-1], qr["p_I"][:-1], color='k', alpha=.8)
        ax1[-1].plot(er["t"][:-1], er["p_I"][:-1], color='r', alpha=.8)
        _ = [x.set_ylim(0, N_pop) for x in ax1[:-1]]
    plt.show()
