from SBM_Routines.Path_Config import *
from SBM_Routines.SBM_Optimization import *
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def read_trajectories(graph_dir):
    trajectories = []
    #get all files that begins with "community"
    for f in os.listdir(graph_dir):
        if f.startswith("community"):
            trajectories.append(np.genfromtxt(graph_dir + f, delimiter=","))

    return trajectories

def read_total_trajectories(graph_dir):
    community_traj = read_trajectories(graph_dir + "/Trajectories/")
    N_communities = int(community_traj[0].shape[1]/3)
    #sum every third column
    total_traj = np.zeros((len(community_traj), community_traj[0].shape[0], 3))
    for i, traj in enumerate(community_traj):
        for j in range(3):
            total_traj[i,:,j] += np.sum(traj[:, j::3], axis=1)
    return total_traj


def plot_SIR_percentile_trajectory(graph_dir,  ax):
    with open(graph_dir+ "../../Sim_Param.json") as f:
        params = json.load(f)
    trajectories = read_total_trajectories(graph_dir)
    mean = np.mean(trajectories, axis=0)
    # std = np.std(trajectories, axis=0)
    #get percentiles
    percentiles = [np.percentile(trajectories, p, axis=0) for p in np.linspace(5, 95, 11)]


    Nt = mean.shape[0]
    #ax is 3 subplots
    #fill_between
    mid_idx= int(floor(len(percentiles)/2))
    for j in range(mid_idx):
        ax.fill_between(np.arange(Nt), percentiles[mid_idx - j][:,1], percentiles[mid_idx + j][:,1], color='gray', alpha=0.1)
    ax.plot(np.arange(Nt), mean[:,1], color='k', linewidth=2, linestyle='--')


if __name__ == '__main__':
    fig, ax = plt.subplots(1,1)

    sim_type = "/Excitation/"
    p_dirs = get_p_dirs("/home/man/Documents/ER_Bernoulli_Robust_MPC/build/data/SIR_sim/")
    p_dirs = [pd + sim_type for pd in p_dirs]
    p_dir = p_dirs[0]
    #json load sim_params
    with open(p_dir + "/Sim_Param.json") as f:
        sim_params = json.load(f)
    graph_dirs = get_graph_dirs(p_dir)
    d_dirs = get_detected_dirs(p_dir)
    d_dir = d_dirs[0]
    plot_SIR_percentile_trajectory(d_dir, ax)
    # plot_LS_predictions(d_dir, ax, sim_params["Nt"], sim_params["N_sims"])
    ax.grid()
    plt.show()
