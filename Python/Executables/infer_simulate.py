from SBM_Routines.Path_Config import *
from SBM_Routines.Structural_Inference import inference_over_p_out, complete_graph_max_edges, flatten_sublists, project_mapping
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
matplotlib.use('TkAgg')


if __name__ == '__main__':

    Np = 10
    q = sycl_queue(cpu_selector())

    N_pop = 100
    N_communities = 10
    p_in = 1.0
    p_out = np.linspace(0.07, 0.16, Np)
    p_in = 0.1
    Gs = []
    states = []
    N_blocks = []
    entropies = []
    fig, ax = plt.subplots(2)
    seeds = np.random.randint(0, 100000, Np)
    pool = mp.Pool(int(mp.cpu_count()/2))

    edgelists, vertex_lists, N_blocks, entropies, vcms, sim_params = inference_over_p_out(N_pop, N_communities, p_in, p_out, seeds, Np)
    block_df = pd.DataFrame(np.array(N_blocks).T, columns=p_out)
    ent_df = pd.DataFrame(np.array(entropies).T, columns=p_out)
    for elist, vcm, p in zip(edgelists, vcms, sim_params):
        vcm_0 = [v[1] for v in vcm]
        vcm = [project_mapping(v[0], v[1])for v in vcm]
        p.N_communities = max([max(v) for v in vcm]) + 1
        run(q, p, elist, vcm)
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
