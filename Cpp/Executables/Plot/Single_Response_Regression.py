# Plot ../data/traj_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys
from sklearn.cross_decomposition import PLSRegression

BINDER_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/build/Binders/"
sys.path.insert(0, BINDER_DIR)
from pyFROLS import Polynomial_Model, Quantile_Regressor, ERR_Regressor

# import pyFROLS as pf
if __name__ == '__main__':
    # read and plot all csv in ../data
    # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    cwd = os.path.dirname(os.path.realpath(__file__))

    DATA_DIR = cwd + '/../../data/'
    FIGURE_DIR = cwd + '/../../figures/'
    csv_model = DATA_DIR + '/model.csv'

    # find all csv in data_path
    files = glob.glob(DATA_DIR + "Bernoulli_SIR_MC_*_60_*1.0*.csv")
    # sort q_files according to float in name
    fig, ax = plt.subplots(4)
    N_rows = 20
    N_dfs = 100
    dfs = [pd.read_csv(f, delimiter=",") for f in files[:N_dfs]]
    step = 1
    df_data = [df[["S", "I", "R", "p_I"]].to_numpy() for df in dfs if df.shape[0] == N_rows]
    X = np.vstack([mat[:-1, :3][::step, :] for mat in df_data])
    Y = np.vstack([mat[1:, :3][::step, :] for mat in df_data])
    U = np.vstack([mat[:-1, 3][::step, np.newaxis] for mat in df_data])



    Nx = X.shape[1]
    Nt = df_data[0][::step,0].shape[0]
    Nu = U.shape[1]
    d_max = 1
    N_output_features = 16
    ERR_tol = 1e-3
    model = Polynomial_Model(Nx, Nu, N_output_features, d_max)
    MAE_tol = 1
    tau = .95
    # regressor = ERR_Regressor(ERR_tol)
    regressor = Quantile_Regressor(tau, MAE_tol)
    regressor.transform_fit(X, U, Y, model)
    x0 = np.mean(X[::Nt-1,:], axis=0)
    u0 = U[0, :]
    U_mean = U[:Nt-1, :]
    # U_mean = np.mean(U.reshape(Nt - 1, -1), axis=1)
    model.feature_summary()
    model.write_csv(DATA_DIR + "/model.csv")
    t = dfs[0]["t"].to_numpy()[::step]
    X_traj = model.simulate(x0, U_mean, Nt - 1)
    print(dfs[0].head())
    for i, letter in enumerate(['S', 'I', 'R', 'p_I']):
        if (i < 3):
            ax[i].plot(t, X_traj[:, i], color='r')
        [ax[i].plot(df["t"], df[letter], color='gray', alpha=.2) for df in dfs[:N_dfs]]
    _ = [x.set_ylim(0, 60) for x in ax[:-1]]
    plt.show()
