from SBM_Routines.Path_Config import *
import json
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns
import pandas as pd
def basename(path):
    return os.path.basename(os.path.normpath(path))

def get_SBM_connection_elements(N_communities, connection_data, is_in_connections):
    ccm = complete_ccm(N_communities, True)
    result = []
    for i, c in enumerate(ccm):
        if ((c[0] == c[1]) and is_in_connections):
            result.append(connection_data[:,i])
        if ((c[0] != c[1]) and not is_in_connections):
            result.append(connection_data[:,i])
    return np.array(result)

def get_beta_std(graph_beta, N_communities, is_in_connection):
    connection_betas = get_SBM_connection_elements(N_communities, graph_beta, is_in_connection)
    return np.std(connection_betas,axis=0)
def get_beta_mean(graph_beta, N_communities, is_in_connection):
    connection_betas = get_SBM_connection_elements(N_communities, graph_beta, is_in_connection)
    return np.mean(connection_betas, axis=0)


if __name__ == '__main__':
    fig, ax = plt.subplots()

    p_dirs = get_p_dirs(Data_dir)
    beta_LS_mean, beta_QR_mean, beta_LS_std, beta_QR_std = [], [], [], []
    for i, p_dir in enumerate(p_dirs):
        #get parameters from jsjon
        params = []
        with open(p_dir + "/Sim_Param.json") as f:
            params = json.load(f)
        graph_dirs = get_graph_dirs(p_dir)

        N_connections = np.genfromtxt(graph_dirs[0] + "/theta_LS.csv", delimiter=",").shape[0]
        graph_beta_LS = np.zeros((params["N_graphs"], N_connections))
        graph_beta_QR = np.zeros((params["N_graphs"], N_connections))
        for g_idx, g_dir in enumerate(graph_dirs):
            graph_beta_LS[g_idx, :] = np.genfromtxt(g_dir + "/theta_LS.csv", delimiter=",")
            graph_beta_QR[g_idx,:] = np.genfromtxt(g_dir + "/theta_QR.csv", delimiter=",")
        beta_LS_mean.append(np.mean(graph_beta_LS, axis=0))
        beta_QR_mean.append(np.mean(graph_beta_QR, axis=0))
        beta_LS_std.append(np.std(graph_beta_LS, axis=0))
        beta_QR_std.append(np.std(graph_beta_QR, axis=0))


    beta_LS_mean = np.array(beta_LS_mean).T
    beta_QR_mean = np.array(beta_QR_mean).T
    beta_LS_std = np.array(beta_LS_std).T
    beta_QR_std = np.array(beta_QR_std).T
    #extract numbers from p_dirs
    p_out = [float(basename(p_dir).split("_")[2]) for p_dir in p_dirs]
    #sort ascending
    p_out = np.sort(p_out)
    LS_mean_df = pd.DataFrame(beta_LS_mean, columns=p_out)
    QR_mean_df = pd.DataFrame(beta_QR_mean, columns=p_out)
    LS_std_df = pd.DataFrame(beta_LS_std, columns=p_out)
    QR_std_df = pd.DataFrame(beta_QR_std, columns=p_out)
    fig, ax = plt.subplots(2,2)
    sns.violinplot(LS_mean_df, ax=ax[0][0])
    sns.violinplot(QR_mean_df, ax=ax[1][0])
    sns.violinplot(LS_std_df, ax=ax[0][1])
    sns.violinplot(QR_std_df, ax=ax[1][1])
    ax[1][0].set_xticklabels([str(round(p, 2)) for p in p_out])
    ax[1][1].set_xticklabels([str(round(p, 2)) for p in p_out])
    ax[1][0].set_xlabel("p_out")
    ax[1][1].set_xlabel("p_out")
    ax[0][0].set_ylabel(r'$\beta_{LS}$')
    ax[1][0].set_ylabel(r'$\beta_{QR}$')
    fig.savefig(Data_dir + "/Regression_Betas.pdf", format='pdf', dpi=1000)


    LS_mean_out_df = pd.DataFrame(get_SBM_connection_elements(N_communities, beta_LS_mean, columns=p_out)
    QR_mean_out_df = pd.DataFrame(beta_QR_mean, columns=p_out)
    LS_std_out_df = pd.DataFrame(beta_LS_std, columns=p_out)
    QR_std_out_df = pd.DataFrame(beta_QR_std, columns=p_out)
