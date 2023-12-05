from SBM_Routines.Path_Config import *
import json

def get_p_Is(d_dir, subdirname, Nt_per_u = 7):
    ccm = np.genfromtxt(d_dir + "/ccm.csv", delimiter=",", dtype=int)
    N_connections = ccm.shape[0]
    p_I = np.genfromtxt(d_dir  + "/" + subdirname + "/u_opt.csv", delimiter=" ")
    if (len(p_I.shape) == 1):
        p_I = np.repeat([p_I], N_connections, axis=0).T
    p_I = np.repeat(p_I, Nt_per_u, axis=0)
    return p_I
def get_edges(fname):
    edges = []
    mat = np.genfromtxt(fname, delimiter=",", dtype=int)
    for i in range(mat.shape[0]):
        edges.append((int(mat[i,0]), int(mat[i,1])))
    return edges

def validate_for_solution(q, base_dir, subdirname, regression_type):
    gdirs = get_graph_dirs(base_dir)
    edgelists, vpms, p_Is = [], [], []
    SBM_Database::Sim_Param = Sim_Param(base_dir + "/Sim_Param.json")
    for gdir in gdirs:
        edgelists.append(get_edges(gdir + "/" + subdirname + "/edgelist.csv"))
        vpms.append(np.genfromtxt(gdir + "/" + subdirname + "/vpm.csv", dtype=int))
        p_Is.extend([get_p_Is(gdir + subdirname, regression_type)]*sim_param.N_sims)

    p_I_run(q, sim_param, edgelists, vpms, p_Is)

if __name__ == '__main__':
    Nt_per_u = 7
    p_dirs = get_p_dirs(Data_dir)
    q = sycl_queue(cpu_selector())
    for p_dir in p_dirs:
        validate_for_solution(q, p_dir, "Detected_Communities", "LS")
        validate_for_solution(q, p_dir, "Detected_Communities", "QR")
        validate_for_solution(q, p_dir, "True_Communities", "LS")
        validate_for_solution(q, p_dir, "True_Communities", "QR")
