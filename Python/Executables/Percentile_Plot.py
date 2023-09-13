from SBM_Routines.Path_Config import *
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
    community_traj = read_trajectories(graph_dir)
    N_communities = int(community_traj[0].shape[1]/3)
    #sum every third column
    total_traj = np.zeros((len(community_traj), community_traj[0].shape[0], 3))
    for i, traj in enumerate(community_traj):
        for j in range(3):
            total_traj[i,:,j] += np.sum(traj[:, j::3], axis=1)
    return total_traj


def plot_SIR_percentile_trajectory(graph_dir, ax):
    with open(graph_dir + "../Sim_Param.json") as f:
        params = json.load(f)
    trajectories = read_total_trajectories(graph_dir)
    mean = np.mean(trajectories, axis=0)
    # std = np.std(trajectories, axis=0)
    #get percentiles
    percentiles = [np.percentile(trajectories, p, axis=0) for p in np.linspace(5, 95, 11)]

    Nt = mean.shape[0]
    #ax is 3 subplots
    #fill_between
    for perc in percentiles:
        for i in range(3):
            ax[i].fill_between(np.arange(Nt), mean[:,i] - perc[:,i], mean[:,i] + perc[:,i], color='gray', alpha=0.05)
    for i in range(3):
        ax[i].plot(np.arange(Nt), mean[:,i], color='k', linewidth=2, linestyle='--')


if __name__ == '__main__':
    fig, ax = plt.subplots(3,1)
    p_dirs = get_p_dirs(Data_dir)
    p_dir = p_dirs[0]
    graph_dirs = get_graph_dirs(p_dir)
    graph_dir = graph_dirs[0]
    plot_SIR_percentile_trajectory(graph_dir, ax)

    plt.show()
