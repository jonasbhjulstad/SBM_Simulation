from SBM_Routines.Path_Config import *
from SBM_Routines.Structural_Inference import inference_over_p_out, complete_graph_max_edges, flatten_sublists, project_mapping
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import json
matplotlib.use('TkAgg')


if __name__ == '__main__':

    Np = 10
    q = sycl_queue(cpu_selector())

    N_communities = 2
    N_graphs = 1
    # p_out = np.linspace(0.07, 0.16, Np)
    # p_out = np.linspace(0.3, 0.7, Np)
    p_out = [0.005]
    p_in = 0.3
    Gs = []
    states = []
    N_blocks = []
    entropies = []
    fig, ax = plt.subplots(2)
    seeds = np.random.randint(0, 100000, Np)
    pool = mp.Pool(int(mp.cpu_count()/2))

    #if Data_dir does not exist, create it
    if not os.path.exists(Data_dir):
        os.makedirs(Data_dir)

    edgelists, vertex_lists, N_blocks, entropies, vcms, sim_params = inference_over_p_out(N_communities, N_communities, p_in, p_out, seeds, N_graphs)
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
