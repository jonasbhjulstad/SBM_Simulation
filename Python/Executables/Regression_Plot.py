from SBM_Routines.Path_Config import *
import json
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import sys
import os
import seaborn as sns
import pandas as pd


def basename(path):
    return os.path.basename(os.path.normpath(path))


def get_SBM_connection_elements(ccm, connection_data, is_in_connections):
    result = []
    if len(ccm) == 1:
        if is_in_connections:
            return np.array([connection_data[0]])
        else:
            return np.array([])
    for i, c in enumerate(ccm):
        if ((c[0] == c[1]) and is_in_connections):
            result.append(connection_data[i])
        if ((c[0] != c[1]) and not is_in_connections):
            result.append(connection_data[i])
    return np.array(result)


def get_SBM_in_connections(ccm, connection_data):
    return get_SBM_connection_elements(ccm, connection_data, True)


def get_SBM_out_connections(ccm, connection_data):
    return get_SBM_connection_elements(ccm, connection_data, False)


def get_beta_std(graph_beta, ccm, is_in_connection):
    connection_betas = get_SBM_connection_elements(
        ccm, graph_beta, is_in_connection)
    return np.std(connection_betas, axis=0)


def get_beta_mean(graph_beta, ccm, is_in_connection):
    connection_betas = get_SBM_connection_elements(
        ccm, graph_beta, is_in_connection)
    return np.mean(connection_betas, axis=0)


def get_graph_theta_mean_std(p_dir, subdirname, regression_type):
    graph_dirs = get_graph_dirs(p_dir)
    sim_dirs = [gd + "/" + subdirname for gd in graph_dirs]
    beta_LS_mean, beta_QR_mean, beta_LS_std, beta_QR_std = [], [], [], []
    N_graphs = len(sim_dirs)
    theta_outs = []
    theta_ins = []
    for g_idx, g_dir in enumerate(sim_dirs):
        theta = to_1D(np.genfromtxt(g_dir + "/theta_" +
                      regression_type + ".csv", delimiter=","))
        ccm = to_2D(np.genfromtxt(g_dir + "/ccm.csv", delimiter=","))
        theta_ins.extend(get_SBM_in_connections(ccm, theta))
        theta_outs.extend(get_SBM_out_connections(ccm, theta))
    theta_ins = np.array(theta_ins)
    theta_outs = np.array(theta_outs)
    theta_ins[np.isnan(theta_ins)] = np.mean(theta_ins[~np.isnan(theta_ins)])
    theta_outs[np.isnan(theta_outs)] = np.mean(
        theta_outs[~np.isnan(theta_outs)])
    # remove all over 100
    theta_ins = theta_ins[theta_ins < 1000]
    theta_outs = theta_outs[theta_outs < 1000]

    return np.mean(theta_ins), np.mean(theta_outs), np.std(theta_ins), np.std(theta_outs)


def get_theta_mean_std(p_dirs, subdirname, regression_type):
    theta_in_mean, theta_in_std, theta_out_mean, theta_out_std = [], [], [], []
    for p_dir in p_dirs:
        theta_in_m, theta_out_m, theta_in_s, theta_out_s = get_graph_theta_mean_std(
            p_dir, subdirname, regression_type)
        theta_in_mean.append(theta_in_m)
        theta_out_mean.append(theta_out_m)
        theta_in_std.append(theta_in_s)
        theta_out_std.append(theta_out_s)
    return theta_in_mean, theta_out_mean, theta_in_std, theta_out_std


def get_parent_dir(p_dir):
    this_dir = os.path.dirname(p_dir)
    return os.path.dirname(this_dir)


def beta_plot_type(p_dirs, subdirname, regression_type, ax, ecolor, transform):
    beta_in_mean, beta_out_mean, beta_in_std, beta_out_std = get_theta_mean_std(
        p_dirs, subdirname, regression_type)
    # get parent_dir
    parent_dirs = [get_parent_dir(p_dir) for p_dir in p_dirs]
    p_out = [float(basename(p_dir).split("_")[-1]) for p_dir in parent_dirs]
    ax[0].errorbar(p_out, beta_in_mean, yerr=beta_in_std, fmt='.', color=ecolor,
                   ecolor=ecolor, label=regression_type, capsize=4, transform=transform[0])
    ax[1].errorbar(p_out, beta_out_mean, yerr=beta_out_std, fmt='.',
                   color=ecolor, ecolor=ecolor, capsize=4, label=regression_type, transform=transform[1])


def beta_plot(p_dirs, subdirname):
    fig, ax = plt.subplots(2)
    parent_dirs = [get_parent_dir(p_dir) for p_dir in p_dirs]
    p_out = [float(basename(p_dir).split("_")[-1]) for p_dir in parent_dirs]
    LS_trans = [Affine2D().translate(-0.01, 0.0) + ax[0].transData,
                Affine2D().translate(-0.01, 0.0) + ax[1].transData]
    QR_trans = [Affine2D().translate(0.01, 0.0) + ax[0].transData,
                Affine2D().translate(0.01, 0.0) + ax[1].transData]

    beta_plot_type(p_dirs, subdirname, "LS", ax, 'k', LS_trans)
    beta_plot_type(p_dirs, subdirname, "QR", ax, 'r', QR_trans)
    # set xticks to be at p_out
    ax[0].set_xticks(p_out)
    ax[1].set_xticks(p_out)

    ax[1].set_xlabel("p_out")
    ax[0].set_ylabel(r'$\hat{\theta}_{in}$')
    ax[1].set_ylabel(r'$\hat{\theta}_{out}$')
    ax[0].legend()
    [x.grid(True) for x in ax]
    fig.savefig(Data_dir + "/Regression_Betas.pdf", format='pdf', dpi=1000)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    p_dirs = get_p_dirs(
        "/home/man/Documents/ER_Bernoulli_Robust_MPC/build/data/SIR_sim/")

    excitation_p_dirs = [pd + "/Excitation/" for pd in p_dirs]
    validation_p_dirs = [pd + "/Validation/" for pd in p_dirs]

    beta_plot(excitation_p_dirs, "Detected_Communities")

    # extract numbers from p_dirs
    p_out = [float(basename(p_dir).split("_")[2]) for p_dir in p_dirs]
    # sort ascending
    p_out = np.sort(p_out)
    # sort p_dirs wrt p_out
    p_dirs = [p_dir for _, p_dir in sorted(zip(p_out, p_dirs))]

    # LS_mean_out_df = pd.DataFrame(get_SBM_connection_elements(, beta_LS_mean, columns=p_out)
    # QR_mean_out_df = pd.DataFrame(beta_QR_mean, columns=p_out)
    # LS_std_out_df = pd.DataFrame(beta_LS_std, columns=p_out)
    # QR_std_out_df = pd.DataFrame(beta_QR_std, columns=p_out)
