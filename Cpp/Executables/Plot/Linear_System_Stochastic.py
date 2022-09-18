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
   # read and plot all csv in ../data
    # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    cwd = os.path.dirname(os.path.realpath(__file__))


    DATA_DIR = cwd + '/../../data/'
    FIGURE_DIR = cwd + '/../../figures/'


        

    # find all csv in data_path
    files = glob.glob(DATA_DIR + "Linear_Stochastic_Traj*.csv")
    qr_file = glob.glob(DATA_DIR + "Linear_Stochastic_QR.csv")
    er_file = glob.glob(DATA_DIR + "Linear_Stochastic_ERR.csv")
    #sort q_files according to float in name
    fig, ax = plt.subplots(4)
    X_trajs = [np.genfromtxt(f, delimiter=",") for f in files]

    qr_df = pd.read_csv(qr_file[0])
    er_df = pd.read_csv(er_file[0])

    for X in X_trajs:
        ax[0].plot(qr_df["t"], X[:, 0], color='gray', alpha=.2)
        ax[1].plot(qr_df["t"], X[:, 1], color='gray', alpha=.2)
        ax[2].plot(qr_df["t"], X[:, 2], color='gray', alpha=.2)
    ax[0].plot(qr_df["t"], qr_df["S"], color='k', alpha=.8)
    ax[1].plot(qr_df["t"], qr_df["I"], color='k', alpha=.8)
    ax[2].plot(qr_df["t"], qr_df["R"], color='k', alpha=.8)
    ax[3].plot(qr_df["t"], qr_df["u1"], color='k', alpha=.8)
    ax[0].plot(er_df["t"], er_df["S"], color='r', alpha=.8)
    ax[1].plot(er_df["t"], er_df["I"], color='r', alpha=.8)
    ax[2].plot(er_df["t"], er_df["R"], color='r', alpha=.8)
    ax[3].plot(er_df["t"], er_df["u0"], color='r', alpha=.8)
    # _ = [x.set_ylim(0,60) for x in ax]
    # plot S, I, R, p_I, p_R
    plt.show()
