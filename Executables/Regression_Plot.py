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
Data_dir = Project_root + "data/"
Graphs_dir = Data_dir + "SIR_sim/"

def basename(path):
    return os.path.basename(os.path.normpath(path))


if __name__ == '__main__':
    fig, ax = plt.subplots()

    #find all dirs that starts with p_out
    dirs = os.listdir(Graphs_dir)
    p_dirs = [Graphs_dir + d for d in dirs if basename(d).startswith("p_out")]
    for i, p_dir in enumerate(p_dirs):
        #get parameters from jsjon
        params = []
        with open(p_dir + "/Sim_Param.json") as f:
            params = json.load(f)

        graph_dirs = os.listdir(p_dir)
        graph_dirs = [p_dir + "/" + d for d in graph_dirs if d.startswith("Graph")]

        N_connections = np.genfromtxt(graph_dirs[0] + "/theta_LS.csv", delimiter=",").shape[0]
        beta_LS = np.zeros((params["N_graphs"], N_connections))
        beta_QR = np.zeros((params["N_graphs"], N_connections))
        for g_idx, g_dir in enumerate(graph_dirs):
            beta_LS[g_idx, :] = np.genfromtxt(g_dir + "/theta_LS.csv", delimiter=",")
            beta_QR[g_idx,:] = np.genfromtxt(g_dir + "/theta_QR.csv", delimiter=",")
        a = 1

    p_outs = np.linspace(0.0, 0.08, params["N_graphs"])
