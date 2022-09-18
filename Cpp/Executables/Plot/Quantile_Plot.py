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
    if (len(sys.argv) <= 1):
        network_type = "SIS"
        statenames = ["S", "I"]
    else:
        network_type = "SIR"
        statenames = ["S", "I", "R"]
        # read and plot all csv in ../data
        # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    cwd = os.path.dirname(os.path.realpath(__file__))

    DATA_DIR = cwd + '/../../data/'
    FIGURE_DIR = cwd + '/../../figures/'

    # find all csv in data_path
    files = glob.glob(DATA_DIR + "Bernoulli_" + network_type + "_MC_*60_1.000000.csv")
    qr_file = glob.glob(DATA_DIR + "Quantile_Trajectory_" + network_type + ".csv")
    er_file = glob.glob(DATA_DIR + "ERR_Trajectory_" + network_type + ".csv")
    # sort q_files according to float in name
    fig, ax = plt.subplots(len(statenames) + 1)
    dfs = [pd.read_csv(f, delimiter=",") for f in files[:500]]

    qr_df = pd.read_csv(qr_file[0])
    er_df = pd.read_csv(er_file[0])

    I = np.zeros_like(dfs[0]["S"].to_numpy())
    for j, df in enumerate(dfs):
        X = df[statenames].to_numpy()
        for i, name in enumerate(statenames):
            ax[i].plot(df["t"], df[name], color='gray', alpha=.2)
        ax[-1].plot(df["t"][:-1], df["p_I"][:-1], color='gray', alpha=.2)
        if (np.any(df["t"] > 101)):
            a = 1
    for i, name in enumerate(statenames):
        ax[i].plot(qr_df["t"][:-1], qr_df[name][:-1], color='k', alpha=.8)
        ax[i].plot(er_df["t"][:-1], er_df[name][:-1], color='r', alpha=.8)
    ax[-1].plot(qr_df["t"][:-1], qr_df["p_I"][:-1], color='k', alpha=.8)
    ax[-1].plot(er_df["t"][:-1], er_df["p_I"][:-1], color='r', alpha=.8)
    _ = [x.set_ylim(0, 60) for x in ax[:-1]]
    # plot S, I, R, p_I, p_R
    plt.show()
