#Plot ../data/traj_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

if __name__ == '__main__':
   #read and plot all csv in ../data
    data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    #find all csv in data_path
    files = glob.glob(data_path + "Bernoulli_SIR_MC_*.csv")

    fig, ax = plt.subplots(5)

    #plot S, I, R, p_I, p_R
    for i, file in enumerate(files[::100]):
        df = pd.read_csv(file)
        ax[0].plot(df['S'], color='k', label = file)
        ax[1].plot(df['I'], color='k', label = file)
        ax[2].plot(df['R'], color='k', label = file)
        ax[3].plot(df['p_I'], color='k', label = file)
        ax[4].plot(df['p_R'], color='k', label = file)
    plt.show()
