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
    csv_model = DATA_DIR + '/model.csv'


        

    # find all csv in data_path
    files = glob.glob(DATA_DIR + "Bernoulli_SIR_MC_1000_1_*.csv")
    #sort q_files according to float in name
    fig, ax = plt.subplots(4)
    dfs = [pd.read_csv(f, delimiter=",") for f in files]

    I = np.zeros_like(dfs[0]["S"].to_numpy())
    for df in dfs:
        X = df[["S", "I", "R"]].to_numpy()
        ax[0].plot(df["t"], df["S"], color='gray', alpha=.2)
        ax[1].plot(df["t"], df["I"], color='gray', alpha=.2)
        ax[2].plot(df["t"], df["R"], color='gray', alpha=.2)
        ax[3].plot(df["t"], df["p_I"], color='gray', alpha=.2)


    # plot S, I, R, p_I, p_R
    plt.show()
