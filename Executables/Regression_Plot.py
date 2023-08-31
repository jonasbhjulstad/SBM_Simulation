import json
from collections import Counter
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import sys
import graph_tool.all as gt
import os
import seaborn as sns
from itertools import repeat

Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
sys.path.append(Binder_path)
from SIR_SBM import *
Data_dir = Project_root + "/data/SIR_sim/"
Graphs_dir = Data_dir + "Graphs/"



if __name__ == '__main__':
    fig, ax = plt.subplots()

    N_graphs = 10
    p_outs = np.linspace(0.0, 0.08, N_graphs)
    #find all dirs that starts with p_out
    dirs = os.listdir(Data_dir)
    dirs = [d for d in dirs if d.startswith("p_out")]
    for i, p_dir in enumerate(dirs):
        graph_dirs = os.listdir(Data_dir + p_dir + "/")
        graph_dirs = [Data_dir + "/" + p_dir + "/" + d for d in graph_dirs if d.startswith("Graph")]
        N_connections = np.genfromtxt(graph_dirs[0] + "/theta_LS.csv", delimiter=",").shape[0]
        assert len(graph_dirs) == N_graphs
        beta_LS = np.zeros((N_graphs, N_connections))
        beta_QR = np.zeros((N_graphs, N_connections))
        for g_idx, g_dir in enumerate(graph_dirs):
            beta_LS[g_idx, :] = np.genfromtxt(g_dir + "/theta_LS.csv", delimiter=",")
            beta_QR[g_idx,:] = np.genfromtxt(g_dir + "/theta_QR.csv", delimiter=",")

        a = 1
