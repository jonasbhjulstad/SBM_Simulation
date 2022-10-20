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

# def filename_sort(filenames):
#     files = {}
#     for name in filenames:
#         # find all numbers in the file name
#         p_ER = re.findall(r'\d+\.\d+', name)[0]
#         N_pop = basename(name).split('_')[4]
#         if (N_pop not in files.keys()):
#             files[N_pop] = {}
#         if (p_ER not in files[N_pop].keys()):
#             files[N_pop][p_ER] = []
#         files[N_pop][p_ER].append(name)
#     return files


if __name__ == '__main__':
   # read and plot all csv in ../data
    # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
   
    # find all csv in data_path
    p_I0 = 1.0
    N_pop = 50
    fnames = glob.glob(DATA_DIR + "Quantile_Bernoulli_SIR_MC_" + str(N_pop) + "_1/*.csv")
    # pd read first line
    #sort fnames according to int of csv file name
    fnames_sorted = sorted(fnames, key=lambda x: int(os.path.basename(x)[:-4]))


    # def read_p_I(fname):
    #     df = pd.read_csv(fname, delimiter=",", nrows=1)
    #     return df["p_I"][0]

    # p_Is = pool.map(read_p_I, files[:100])
    # sort q_files according to float in name
    quantiles = []
    quantiles = [pd.read_csv(fname, delimiter = ',') for fname in fnames_sorted]
    

    #plot quantiles
    fig, ax = plt.subplots(3)
    for q in quantiles:
        ax[0].plot(q["t"], q["S"], color='gray', alpha=.2)
        ax[1].plot(q["t"], q["I"], color='gray', alpha=.2)
        ax[2].plot(q["t"], q["R"], color='gray', alpha=.2)
    plt.show()
