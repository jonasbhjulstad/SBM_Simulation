from SBM_Routines.Path_Config import *
from SBM_Routines.Structural_Inference import inference_over_p_out, complete_graph_max_edges, flatten_sublists, project_mapping, multiple_structural_inference_over_p_out
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import json
matplotlib.use('TkAgg')

def get_p_I(R0, N_pop, dt = 1):
    alpha = 0.1
    return 1 - np.exp(-R0*alpha*dt/N_pop*10)
def create_sim_param(N_communities, p_in, p_out, N_pop, seed, N_graphs, R0_min, R0_max):
    p = Sim_Param()
    p.N_communities = N_communities
    p.N_pop = N_pop
    p.N_sims = 100
    p.p_in = p_in
    p.p_out = p_out
    p.Nt = 30
    p.Nt_alloc = 30
    p.p_R0 = 0.0
    p.p_R = 0.1
    p.p_I0 = 0.1
    p.p_I_min = get_p_I(R0_min, p.N_pop)
    p.p_I_max = get_p_I(R0_max, p.N_pop)
    p.seed = seed
    p.N_graphs = N_graphs
    p.simulation_subdir = "/Detected_Communities/"
    p_out_str = str(p_out)[:4]
    if len(p_out_str) < 4:
        p_out_str += "0"*(4-len(p_out_str))

    p.output_dir = Data_dir + "p_out_" + p_out_str + "/"
    p.compute_range=sycl_range_1(p.N_graphs*p.N_sims)
    p.wg_range=sycl_range_1(p.N_graphs*p.N_sims)
    return p


if __name__ == '__main__':

    Np = 10
    q = sycl_queue(cpu_selector())

    N_communities = 10
    N_graphs = 1
    # p_out = np.linspace(0.07, 0.16, Np)
    p_out = np.linspace(0.05,0.15, Np)[:1]
    # p_out = [0.005]
    p_in = 0.1
    Gs = []
    states = []
    N_blocks = []
    entropies = []
    N_pop = 100
    fig, ax = plt.subplots(2)
    seeds = np.random.randint(0, 100000, Np)
    pool = mp.Pool(int(mp.cpu_count()/2))

    #if Data_dir does not exist, create it
    if not os.path.exists(Data_dir):
        os.makedirs(Data_dir)

    #random number up to uint32 max
    seeds = np.random.randint(0, 100000, Np)
    sim_params = [create_sim_param(N_communities, p_in, po, N_pop, s, N_graphs, 0.5, 2.5) for po, s in zip(p_out, seeds)]
    # N_blocks, entropies = multiple_structural_inference_over_p_out(sim_params)

    edgelists, vertex_lists, N_blocks, entropies, vcms, sim_params = inference_over_p_out(sim_params)
    block_df = pd.DataFrame(np.array(N_blocks).T, columns=p_out)
    ent_df = pd.DataFrame(np.array(entropies).T, columns=p_out)
    #violin plot
    sns.violinplot(block_df, ax=ax[0], cut=0)
    #limit x to 2 decimals
    ax[0].set_xticklabels([str(round(p, 2)) for p in p_out])
    sns.violinplot(ent_df, ax=ax[1], cut=0, scale='width')
    ax[1].set_xticklabels([str(round(p, 2)) for p in p_out])
    ax[1].set_xlabel("p_out")
    ax[0].set_ylabel("Number of blocks")
    ax[1].set_ylabel("'Entropy'/Minimum Description Length")
    fig.savefig(Data_dir + "/Structural_Plot.pdf", format='pdf', dpi=1000)
    # plt.show()
    # True_Param.json dump N_communities
    with open(Data_dir + "True_Param.json", 'w') as f:
        json.dump({"N_communities": N_communities}, f)
    for elist, vcm, p in zip(edgelists, vcms, sim_params):
        print("p_out: ", p.p_out)
        base_vcm = [v[0] for v in vcm]
        inferred_vcm = [v[1] for v in vcm]
        detected_vcm = [project_mapping(v[1], v[0])for v in vcm]

        p.N_communities = max([max(v) for v in base_vcm]) + 1
        print("Simulation for true communities")
        p.simulation_subdir = "/True_Communities/"
        run(q, p, elist, base_vcm)

        print("Simulation for inferred communities")
        p.simulation_subdir = "/Detected_Communities/"
        p.N_communities = max([max(v) for v in detected_vcm]) + 1
        run(q, p, elist, detected_vcm)
