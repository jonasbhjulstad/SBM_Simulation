# Plot ../data/traj_

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import glob
import re
from multiprocessing.pool import ThreadPool
from os.path import basename
import os

cwd = os.path.dirname(os.path.realpath(__file__))


DATA_DIR = cwd + '/../../data/'
FIGURE_DIR = cwd + '/../../figures/'

def quantile_plot(ax, q_dfs):
    median_idx = int(len(q_dfs)/2)
    for i in range(median_idx-2):
        lo = q_dfs[median_idx - i]
        hi = q_dfs[median_idx + i]
        a = .7*(1-(i/(median_idx-2)))
        ax[0].fill_between(lo["t"], lo["S"], hi["S"], alpha=a, color='gray')
        ax[1].fill_between(lo["t"], lo["I"], hi["I"], alpha=a, color='gray')
        ax[2].fill_between(lo["t"], lo["R"], hi["R"], alpha=a, color='gray')
    _ = [x.grid() for x in ax]

def filename_sort(filenames):
    files = {}
    for name in filenames:
        # find all numbers in the file name
        p_ER = re.findall(r'\d+\.\d+', name)[0]
        N_pop = basename(name).split('_')[4]
        if (N_pop not in files.keys()):
            files[N_pop] = {}
        if (p_ER not in files[N_pop].keys()):
            files[N_pop][p_ER] = []
        files[N_pop][p_ER].append(name)
    return files


if __name__ == '__main__':
   # read and plot all csv in ../data
    # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
   
    # find all csv in data_path
    p_I0 = 0.2

    fnames = glob.glob(DATA_DIR + "Bernoulli_SIR_MC_Quantiles*.csv")
    # pd read first line
    fnames_sorted = filename_sort(fnames)

    # def read_p_I(fname):
    #     df = pd.read_csv(fname, delimiter=",", nrows=1)
    #     return df["p_I"][0]

    # p_Is = pool.map(read_p_I, files[:100])
    # sort q_files according to float in name
    quantiles = []
    for fname_pop in fnames_sorted.values():
        for fname_p_ER in fname_pop.values():
            fname_p_ER.sort(key=lambda s: float(s.split('_')[-1][:-4]))
            quantiles.append([pd.read_csv(f, delimiter=',') for f in fname_p_ER])
    
    fig= plt.figure(figsize=(10, 10))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    N_pops = list(fnames_sorted.keys())
    N_pops = [N_pops[0],N_pops[0], N_pops[1], N_pops[1]]
    p_ERs = list(list(fnames_sorted.values())[0].keys())
    p_ERs = [p_ERs[0], p_ERs[1], p_ERs[0], p_ERs[1]]

    for i in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        axO = plt.Subplot(fig, outer[i])
        axO.set_title(r'$N_{pop} = ' + str(N_pops[i]) + ', p_{ER} = ' + str(p_ERs[i]) + '$')
        axO.set_yticklabels([])
        axO.set_xticklabels([])
        axO.set_xticks([])
        axO.set_yticks([])
        fig.add_subplot(axO)
        ax = [plt.Subplot(fig, inner[j]) for j in range(3)]
        quantile_plot(ax, quantiles[i])
        # ax.set_ylabel(ylabels[j])
        _ = [x.set_xticklabels([]) for x in ax]
        plt.grid()

        # _ = [x.set_yticklabels([]) for x in ax]
        [fig.add_subplot(x) for x in ax]
    fig.savefig(FIGURE_DIR + "Bernoulli_SIR_MC_Quantiles.svg", format='svg')
    # plot S, I, R, p_I, p_R
    plt.show()
