# Plot ../data/traj_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import os
if __name__ == '__main__':
   # read and plot all csv in ../data
    # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    cwd = os.path.dirname(os.path.realpath(__file__))


    DATA_DIR = cwd + '/../../data/'
    FIGURE_DIR = cwd + '/../../figures/'
    # find all csv in data_path
    files = glob.glob(DATA_DIR + "Bernoulli_SIR_MC_*.csv")
    #sort q_files according to float in name
    fig, ax = plt.subplots(3)
    dfs = [pd.read_csv(f, delimiter=",") for f in files[::100]]

    for df in dfs:
        ax[0].plot(df["t"], df["S"], color='gray', alpha=.2)
        ax[1].plot(df["t"], df["I"], color='gray', alpha=.2)
        ax[2].plot(df["t"], df["R"], color='gray', alpha=.2)

    # plot S, I, R, p_I, p_R
    plt.show()
