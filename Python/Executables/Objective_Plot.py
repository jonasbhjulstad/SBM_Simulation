from SBM_Routines.Path_Config import *
import json
from matplotlib import pyplot as plt
from SBM_Routines.SBM_Optimization import construct_community_ODE, construct_objective_from_ODE, get_objective_value
import numpy as np
import sys
import os
import seaborn as sns
import pandas as pd
def basename(path):
    return os.path.basename(os.path.normpath(path))

def get_SBM_connection_elements(ccm, connection_data, is_in_connections):
    result = []
    for i, c in enumerate(ccm):
        if ((c[0] == c[1]) and is_in_connections):
            result.append(connection_data[:,i])
        if ((c[0] != c[1]) and not is_in_connections):
            result.append(connection_data[:,i])
    return np.array(result)

def get_graph_pred_f(dirname, regtype):
    graph_dirs = get_graph_dirs(dirname)
    f_unif = []
    fs = []
    for gd in graph_dirs:
        f_unif.append(np.genfromtxt(dirname + "/" + regtype + "/" + "f_uniform.csv"))
        fs.append(np.genfromtxt(gd + "/" + regtype + "/" + "f.csv"))
    return np.array(f_unif), np.array(fs)

def get_graph_sim_f(graph_dirname, subdirname,  p, Wu):
    ccmap = np.genfromtxt(graph_dirname + "/" + subdirname + "/ccm.csv", delimiter=",", dtype=int)
    #if len(shape) = 1
    if (len(ccmap.shape) == 1):
        ccmap.resize((ccmap.shape[0], 1))
    ccmap = ccmap[:,:2]
    N_connections = ccmap.shape[0]


    trajs = read_trajectories(graph_dirname + "/" + subdirname + "/Trajectories/")
    p_Is = [to_2D(np.genfromtxt(graph_dirname + "/" + subdirname + "/p_Is/p_I_" + str(i) + ".csv", delimiter=",")) for i in range(p.N_sims)]

    fs = [get_objective_value(traj, p_I, Wu, p.p_I_max) for traj, p_I in zip(trajs, p_Is)]

    return np.array(fs)

def get_p_sim_f(p_dir, subdirname, p, Wu):
    graph_dirs = get_graph_dirs(p_dir)
    fs = []
    for gd in graph_dirs:
        fs.append(get_graph_sim_f(gd,subdirname, p, Wu))
    return np.array(fs)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    p_dirs = get_p_dirs("/home/man/Documents/ER_Bernoulli_Robust_MPC/build/data/SIR_sim/")

    excitation_p_dirs = [pd + "/Excitation/" for pd in p_dirs]
    validation_p_dirs = [pd + "/Validation/" for pd in p_dirs]
    sim_params = [Sim_Param(pd + "/Sim_Param.json") for pd in p_dirs]



    #extract numbers from p_dirs
    p_out = [float(basename(p_dir).split("_")[2]) for p_dir in p_dirs]
    #sort ascending
    p_out = np.sort(p_out)
    #sort p_dirs wrt p_out
    p_dirs = [p_dir for _, p_dir in sorted(zip(p_out, p_dirs))]
    Wu = 1500
    sim_fs_excitation = [get_p_sim_f(pd, "Detected_Communities", p, Wu) for pd, p in zip(excitation_p_dirs, sim_params)]
    sim_fs_validation = [get_p_sim_f(pd, "Detected_Communities", p, Wu) for pd, p in zip(validation_p_dirs, sim_params)]

    sim_fs_excitation_mean = np.mean(sim_fs_excitation, axis=2)
    sim_fs_validation_mean = np.mean(sim_fs_validation, axis=2)
    sim_fs_excitation_std = np.std(sim_fs_excitation, axis=2)
    sim_fs_validation_std = np.std(sim_fs_validation, axis=2)



    graph_fs_excitation_mean = np.mean(sim_fs_excitation_mean, axis=1)
    graph_fs_validation_mean = np.mean(sim_fs_validation_mean, axis=1)
    graph_fs_excitation_std = np.mean(sim_fs_excitation_std, axis=1)
    graph_fs_validation_std = np.mean(sim_fs_validation_std, axis=1)



    fig, ax = plt.subplots(2,2)
    sns.violinplot(pd.DataFrame(sim_fs_excitation_mean, columns=p_out), ax=ax[0][0])
    sns.violinplot(pd.DataFrame(sim_fs_validation_mean, columns=p_out), ax=ax[1][0])
    sns.violinplot(pd.DataFrame(sim_fs_excitation_std, columns=p_out), ax=ax[0][1])
    sns.violinplot(pd.DataFrame(sim_fs_validation_std, columns=p_out), ax=ax[1][1])
    ax[1][0].set_xticklabels([str(round(p, 2)) for p in p_out])
    ax[1][1].set_xticklabels([str(round(p, 2)) for p in p_out])
    ax[1][0].set_xlabel("p_out")
    ax[1][1].set_xlabel("p_out")
    ax[0][0].set_ylabel(r'$\beta_{LS}$')
    ax[1][0].set_ylabel(r'$\beta_{QR}$')
    fig.savefig(Data_dir + "/Regression_Betas.pdf", format='pdf', dpi=1000)


    # LS_mean_out_df = pd.DataFrame(get_SBM_connection_elements(, beta_LS_mean, columns=p_out)
    # QR_mean_out_df = pd.DataFrame(beta_QR_mean, columns=p_out)
    # LS_std_out_df = pd.DataFrame(beta_LS_std, columns=p_out)
    # QR_std_out_df = pd.DataFrame(beta_QR_std, columns=p_out)
