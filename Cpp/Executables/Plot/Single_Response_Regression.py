# Plot ../data/traj_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys
BINDER_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/build/Binders/"
sys.path.insert(0, BINDER_DIR)
from pyFROLS import *
import pyFROLS as ps

# import pyFROLS as pf
if __name__ == '__main__':
   # read and plot all csv in ../data
    # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    cwd = os.path.dirname(os.path.realpath(__file__))


    DATA_DIR = cwd + '/../../data/'
    FIGURE_DIR = cwd + '/../../figures/'
    csv_model = DATA_DIR + '/model.csv'


        

    # find all csv in data_path
    files = glob.glob(DATA_DIR + "Bernoulli_SIR_MC_*60*1.0*.csv")
    #sort q_files according to float in name
    fig, ax = plt.subplots(3)
    N_dfs = 100
    dfs = [pd.read_csv(f, delimiter=",") for f in files[:N_dfs]]
    df_data = [df[["S", "I", "R", "p_I"]].to_numpy() for df in dfs]
    X = np.vstack([mat[:-1, :3] for mat in df_data])
    Y = np.vstack([mat[1:, :3] for mat in df_data])
    U = np.vstack([mat[:-1, 3][:,np.newaxis] for mat in df_data])
    Nx = X.shape[1]
    Nt = X.shape[0]
    Nu = 0
    d_max = 1
    N_output_features = 20
    ERR_tol = 1e-3
    model = Polynomial_Model(N_output_features, d_max)
    model.multiple_response_regression(X, U, Y, ERR_tol)
    model.print()
    x0 = X[0,:]
    u0 = U[0,:]
    Nt = 100
    U = np.ones(Nt)*u0
    model.feature_summary()
    model.write_csv(DATA_DIR + "/model.csv")

    X_traj = model.simulate(x0, U, Nt)

    for i, letter in enumerate(['S', 'I', 'R']):
        ax[i].plot(X_traj[:,i], color='r')
        [ax[i].plot(df[letter], color='gray', alpha=.2) for df in dfs[:N_dfs]]

    plt.show()

