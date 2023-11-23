import sys
import os
import numpy as np
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"  # nopep 8
SBM_Database_path = "~/Documents/SBM_Database/build/binders/"
Data_dir = Project_root + "data/SIR_sim/"  # nopep 8
sys.path.append(SBM_Database_path)  # nopep 8


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_p_dirs(base_dir):
    p_dirs = []
    for d in os.listdir(base_dir):
        if (os.path.isdir(base_dir + d) and d.startswith("p_out")):
            p_dirs.append(base_dir + d + "/")
    # get float number
    p_dirs = sorted(p_dirs, key=lambda x: float(
        x.split("/")[-2].split("_")[-1]))
    return p_dirs


def get_graph_dirs(base_dir):
    graph_dirs = []
    for d in os.listdir(base_dir):
        if (os.path.isdir(base_dir + d) and d.startswith("Graph")):
            graph_dirs.append(base_dir + d + "/")
    # get float number
    graph_dirs = sorted(graph_dirs, key=lambda x: float(
        x.split("/")[-2].split("_")[-1]))
    return graph_dirs


def get_detected_dirs(base_dir):
    graph_dirs = get_graph_dirs(base_dir)
    return [gd + "/Detected_Communities/" for gd in graph_dirs]


def get_true_dirs(base_dir):
    graph_dirs = get_graph_dirs(base_dir)
    return [gd + "/True_Communities/" for gd in graph_dirs]


def read_trajectories(graph_dir):
    trajectories = []
    # get all files that begins with "community"
    for f in os.listdir(graph_dir):
        if f.startswith("community"):
            trajectories.append(np.genfromtxt(graph_dir + f, delimiter=","))

    return trajectories


def to_1D(vec):
    if len(vec.shape) == 0:
        return np.array([vec])
    elif len(vec.shape) == 1:
        return vec
    else:
        return vec.reshape(vec.shape[0])


def to_2D(vec, axis=0):
    if len(vec.shape) == 0:
        return np.array([[vec]])
    elif len(vec.shape) == 1:
        return vec.reshape(1, vec.shape[0])
    else:
        return vec


def N_connections_from_thetas(thetas):
    N_connections = 0
    if (len(thetas.shape) == 0):
        N_connections = 1
        thetas = np.array([thetas])
    else:
        N_connections = thetas.shape[0]
    return N_connections
