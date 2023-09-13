from SBM_Routines.Path_Config import *
import json
from matplotlib import pyplot as plt
import numpy as np
import sys
import os

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
