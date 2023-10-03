from SBM_Routines.Path_Config import *
from SIR_SBM import *
from SBM_Routines.Structural_Inference import inference_over_p_out, complete_graph_max_edges, flatten_sublists, project_mapping, multiple_structural_inference_over_p_out
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import json
matplotlib.use('TkAgg')


# float p_I_from_R0(SIR_State_t state, float R0)
# {

# // return 1-np.exp(np.log(1-p_R)*R0/E_degree*I_proportion)
#     auto E_degree = p_ER*N_pop;
#     // auto I_proportion = static_cast<double>(state.I)/N_pop;
#     // auto I_proportion = 1.f/N_pop;
#     return 1-std::exp(std::log(1-p_R)*R0/E_degree);
# }

def ER_p_I(edgelist, N_pop, R0, p_R=0.1):
    E_degree = max([len(el) for el in edgelist])/(N_pop*2)
    return 1-np.exp(np.log(1-p_R)*R0/E_degree)


def create_sim_param(N_communities, p_in, p_out, N_pop, seed, N_graphs, R0_min, R0_max):
    p = Sim_Param()
    p.N_communities = N_communities
    p.N_pop = N_pop
    p.N_sims = 10
    p.p_in = p_in
    p.p_out = p_out
    p.Nt = 56
    p.Nt_alloc = 30
    p.p_R0 = 0.0
    p.p_R = 0.1
    p.p_I0 = 0.1
    p.p_I_min = 0.01
    p.p_I_max = 0.01
    p.seed = seed
    p.N_graphs = N_graphs
    p_out_str = str(p_out)[:4]
    if len(p_out_str) < 4:
        p_out_str += "0"*(4-len(p_out_str))

    p.output_dir = Data_dir + "p_out_" + p_out_str + "/"
    p.compute_range = sycl_range_1(p.N_graphs*p.N_sims)
    p.wg_range = sycl_range_1(p.N_graphs*p.N_sims)
    return p


def create_multi_param(q, N_communities, p_in, p_out, N_pop, seed, N_graphs):
    p = Multiple_Sim_Param(q)
    p.N_communities = N_communities
    p.N_pop = N_pop
    p.N_sims = 10
    p.p_in = p_in
    p.p_out = p_out
    p.Nt = 56
    p.Nt_alloc = 30
    p.p_R0 = 0.0
    p.p_R = 0.1
    p.p_I0 = 0.1
    p.seed = seed
    p.N_graphs = N_graphs
    p_out_str = str(p_out)[:4]
    if len(p_out_str) < 4:
        p_out_str += "0"*(4-len(p_out_str))

    p.output_dir = Data_dir + "p_out_" + p_out_str + "/"
    p.compute_range = sycl_range_1(len(p_out)*p.N_graphs*p.N_sims)
    p.wg_range = sycl_range_1(len(p_out)*p.N_graphs*p.N_sims)
    return p


if __name__ == '__main__':

    Np = 10
    q = sycl_queue(cpu_selector())

    N_communities = 2
    N_graphs = 2
    # p_out = np.linspace(0.07, 0.16, Np)
    p_out = np.linspace(0.3, 0.8, Np)
    # p_out = [0.005]
    p_in = 0.5
    Gs = []
    states = []
    N_blocks = []
    entropies = []
    N_pop = 100
    N_pop_tot = N_communities*N_pop
    R0_min = 0.1
    R0_max = 2.0
    fig, ax = plt.subplots(2)
    seeds = np.random.randint(0, 100000, Np)
    pool = mp.Pool(int(mp.cpu_count()/2))

    # if Data_dir does not exist, create it
    if not os.path.exists(Data_dir):
        os.makedirs(Data_dir)

    # random number up to uint32 max
    seeds = np.random.randint(0, 100000, Np)

    multi_p = create_multi_param(q,
                                 N_communities, p_in, p_out, N_pop, seeds[0], N_graphs)
    # N_blocks, entropies = multiple_structural_inference_over_p_out(sim_params)

    edgelists, vertex_lists, N_blocks, entropies, vcms, sim_params = inference_over_p_out(
        [multi_p.to_sim_param(i) for i in range(len(p_out))])
    block_df = pd.DataFrame(np.array(N_blocks).T, columns=p_out)
    ent_df = pd.DataFrame(np.array(entropies).T, columns=p_out)
    # violin plot
    sns.violinplot(block_df, ax=ax[0], cut=0)
    # limit x to 2 decimals
    ax[0].set_xticklabels([str(round(p, 2)) for p in p_out])
    sns.violinplot(ent_df, ax=ax[1], cut=0, scale='width')
    ax[1].set_xticklabels([str(round(p, 2)) for p in p_out])
    ax[1].set_xlabel("p_out")
    ax[0].set_ylabel("Number of blocks")
    ax[1].set_ylabel("'Entropy'/Minimum Description Length")
    fig.savefig(Data_dir + "/Structural_Plot.pdf", format='pdf', dpi=1000)
    # # plt.show()
    # # True_Param.json dump N_communities
    with open(Data_dir + "True_Param.json", 'w') as f:
        json.dump({"N_communities": N_communities}, f)

    Graphs_Output_dir = Project_root + "build/data/SIR_sim/"
    create_dir(Graphs_Output_dir)

    for elist, vcm, p in zip(edgelists, vcms, sim_params):
        print("p_out: ", p.p_out)
        base_vcm = [v[0] for v in vcm]
        inferred_vcm = [v[1] for v in vcm]
        detected_vcm = [project_mapping(v[1], v[0])for v in vcm]
        p.p_I_min = ER_p_I(elist, N_pop_tot, R0_min)
        p.p_I_max = ER_p_I(elist, N_pop_tot, R0_max)
        p.N_communities = max([max(v) for v in base_vcm]) + 1

        for g_idx in range(p.N_graphs):
            for community_type in ["/Detected_Communities/", "/True_Communities/"]:
                # for regression_type in ["/LS/", "/QR/"]:
                p_dir = Graphs_Output_dir + community_type + \
                    "/p_out_" + str(p.p_out)[:4] + "/"
                community_dir = p_dir + "Graph_" + \
                    str(g_idx)
                p.dump(p_dir + "/Sim_Param.json")
                create_dir(community_dir)
                np.savetxt(community_dir + "/edgelist.csv",
                           elist[g_idx], delimiter=",", fmt="%i")
                np.savetxt(community_dir + "/base_vcm.csv",
                           base_vcm[g_idx], delimiter=",", fmt="%i")
                np.savetxt(community_dir + "/vcm.csv",
                           detected_vcm[g_idx], delimiter=",", fmt="%i")
    multi_p.dump(Graphs_Output_dir)
    # print("Simulation for true communities")
    # run(q, p, elist, base_vcm)

    # print("Simulation for inferred communities")
    # run(q, p, elist, detected_vcm)
