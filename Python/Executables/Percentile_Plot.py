from SBM_Routines.Path_Config import *
from SBM_Routines.SBM_Optimization import *
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def read_total_trajectories(graph_dir):
    community_traj = read_trajectories(graph_dir + "/Trajectories/")
    N_communities = int(community_traj[0].shape[1]/3)
    # sum every third column
    total_traj = np.zeros((len(community_traj), community_traj[0].shape[0], 3))
    for i, traj in enumerate(community_traj):
        for j in range(3):
            total_traj[i, :, j] += np.sum(traj[:, j::3], axis=1)
    return total_traj


def plot_SIR_percentile_trajectory(excitation_dir, validation_dir, ax):
    with open(excitation_dir + "../../Sim_Param.json") as f:
        params = json.load(f)
    trajectories = read_total_trajectories(validation_dir)
    mean = np.mean(trajectories, axis=0)
    # std = np.std(trajectories, axis=0)
    # get percentiles
    percentiles = [np.percentile(trajectories, p, axis=0)
                   for p in np.linspace(5, 95, 11)]

    Nt = mean.shape[0]
    # ax is 3 subplots
    # fill_between
    mid_idx = int(floor(len(percentiles)/2))
    for j in range(mid_idx):
        for i in range(3):
            ax[i].fill_between(np.arange(Nt), percentiles[mid_idx - j][:, i],
                               percentiles[mid_idx + j][:, i], color='gray', alpha=0.05)
    for i in range(3):
        ax[i].plot(np.arange(Nt), mean[:, i], color='k',
                   linewidth=2, linestyle='dotted')


def get_MX_total_traj(x):
    N_communities = x.shape[1]
    x_tot = [0, 0, 0]
    for i in range(3):
        for j in range(N_communities):
            x_tot[i] += x[3*j + i]
    return horzcat(*x_tot)


def f_ODE(ccm, theta, Nt, N_communities, N_connections, init_state):
    sym_u = MX.sym("u", Nt, N_connections)
    Nt_per_u = 7
    Nu = int(ceil(Nt / Nt_per_u))
    F = construct_community_ODE(ccm, theta, 0.1, N_communities, N_connections)
    state = construct_ODE_trajectory(
        Nt, sym_u, Nt_per_u, F, init_state, N_connections)
    return Function("f_ODE", [sym_u], [horzcat(*state).T])


def plot_LS_predictions(graph_dir, ax, Nt, N_sims):
    alpha = 0.1
    thetas = to_1D(np.genfromtxt(graph_dir + "theta_LS.csv", delimiter=","))
    ccm = to_2D(np.genfromtxt(graph_dir + "ccm.csv",
                delimiter=",", dtype=int))[:, :2]
    # N_communities = np.max((np.max(ccm, axis=1))) + 1
    N_connections = N_connections_from_thetas(thetas)

    x_init = get_avg_init_state(graph_dir, N_sims)
    N_communities = int(x_init.shape[0]/3)
    f = f_ODE(ccm, thetas, Nt, N_communities, N_connections, x_init)
    u_opt_uniform = np.genfromtxt(
        graph_dir + "/LS/u_opt_uniform.csv", delimiter=" ")
    assert np.all(np.isnan(u_opt_uniform) == False)
    Nt_per_u = 7
    u_opt = np.genfromtxt(graph_dir + "/LS/u_opt.csv", delimiter=" ")
    traj = get_total_traj(f(u_opt))
    traj_uniform = get_total_traj(f(u_opt_uniform).full())
    t = np.array(range(Nt+1))

    for i in range(3):
        ax[i].plot(t, traj[:, i], color='k', linewidth=2, linestyle='--')
        # ax[i].plot(t, traj_uniform[:,i], color='k', linewidth=2, linestyle='dotted')
        # scatter instead
        ax[i].scatter(t, traj_uniform[:, i], color='k', s=5)


def plot_QR_predictions(graph_dir, ax, Nt, N_sims):
    thetas = to_1D(np.genfromtxt(graph_dir + "theta_QR.csv", delimiter=","))
    ccm = to_2D(np.genfromtxt(graph_dir + "ccm.csv",
                delimiter=",", dtype=int))[:, :2]
    # N_communities = np.max((np.max(ccm, axis=1))) + 1
    N_connections = N_connections_from_thetas(thetas)

    x_init = get_avg_init_state(graph_dir, N_sims)
    N_communities = int(x_init.shape[0]/3)
    f = f_ODE(ccm, thetas, Nt, N_communities, N_connections, x_init)
    u_opt_uniform = np.genfromtxt(
        graph_dir + "/QR/u_opt_uniform.csv", delimiter=" ")
    assert np.all(np.isnan(u_opt_uniform) == False)
    Nt_per_u = 7
    u_opt = np.genfromtxt(graph_dir + "/QR/u_opt.csv", delimiter=" ")
    traj = get_total_traj(f(u_opt))
    traj_uniform = get_total_traj(f(u_opt_uniform).full())
    t = np.array(range(Nt+1))

    for i in range(3):
        ax[i].plot(t, traj[:, i], color='r', linewidth=2, linestyle='--')
        # ax[i].plot(t, traj_uniform[:,i], color='k', linewidth=2, linestyle='dotted')
        # scatter instead
        ax[i].scatter(t, traj_uniform[:, i], color='r', s=5)


def percentile_plot(excitation_dir, validation_dir, ax, Nt, N_sims):
    plot_SIR_percentile_trajectory(excitation_dir, validation_dir, ax)
    plot_LS_predictions(excitation_dir, ax, Nt, N_sims)
    plot_QR_predictions(excitation_dir, ax, Nt, N_sims)


if __name__ == '__main__':
    fig, ax = plt.subplots(3, 1)
    p_dirs = get_p_dirs(
        "/home/man/Documents/ER_Bernoulli_Robust_MPC/build/data/SIR_sim/")
    p_ex_dirs = [pd + "/Excitation/" for pd in p_dirs]
    p_val_dirs = [pd + "/Validation/" for pd in p_dirs]

    for p_ex_dir, p_val_dir in zip(p_ex_dirs, p_val_dirs):
        with open(p_ex_dir + "/Sim_Param.json") as f:
            sim_params = json.load(f)
        graph_ex_dirs = get_graph_dirs(p_ex_dir)
        d_ex_dirs = get_detected_dirs(p_ex_dir)
        d_val_dirs = get_detected_dirs(p_val_dir)
        for d_ex_dir, d_val_dir in zip(d_ex_dirs, d_val_dirs):
            fig, ax = plt.subplots(3, 1)
            percentile_plot(d_ex_dir, d_val_dir, ax,
                            sim_params["Nt"], sim_params["N_sims"])
            [x.grid(True) for x in ax]

            ax[0].set_ylabel("$N_{S}$")
            ax[1].set_ylabel("$N_{I}$")
            ax[2].set_ylabel("$N_{R}$")

            # set xlabel to t
            ax[2].set_xlabel("$t$")
            fig.savefig(d_ex_dir + "/Percentile_Plot.png")
            plt.close(fig)
