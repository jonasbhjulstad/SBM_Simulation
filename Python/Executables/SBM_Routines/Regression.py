from SBM_Routines.Path_Config import *
import json
import numpy as np
def regression_over_p_out(base_dir):
    p_dirs = get_p_dirs(base_dir)
    for p_dir in p_dirs:
        sim_param = json.load(open(p_dir + "/Sim_Param.json"))
        graph_dirs = get_graph_dirs(p_dir)
        for g_dir in graph_dirs:
            theta_LS, theta_QR = regression_on_datasets(g_dir, sim_param["N_sims"], sim_param["tau"], 0)
            np.savetxt(g_dir + "/theta_LS.csv", theta_LS)
            np.savetxt(g_dir + "/theta_QR.csv", theta_QR)
