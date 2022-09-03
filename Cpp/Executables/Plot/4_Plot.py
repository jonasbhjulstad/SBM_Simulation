import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import glob
import pandas as pd
if __name__ == '__main__':

    data_path = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/data/"
    N_pops = [20, 50, 20, 50]
    p_ERs = [.1, .1, 1.0, 1.0]
    fnames = glob.glob(data_path + "Bernoulli_SIR_MC_*.csv")

    q_files = glob.glob(data_path + "Quantile_*.csv")
    # sort q_files according to float in name
    q_files.sort(key=lambda s: re.findall("\d+\.\d+", s)[0])
    fig, ax = plt.subplots(4)
    print("Reading quantiles")
    q_dfs = [pd.read_csv(qf, delimiter=",") for qf in q_files]

    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    def percentile_plot(ax, Xi):
        for ci in np.arange(95, 10, -5):
            if ci == 85:
                ax.plot(np.linspace(0,1,Xi.shape[1]), high, color='k', linestyle='dashed', linewidth=1)
            ax.fill_between(np.linspace(0,1,Xi.shape[1]), low, high, color='gray', alpha= .3*(1.0-np.exp(-.01*ci)))
        #disable y axis labels
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        # ax.set_xticks([])
        # ax.set_yticks([])

    ylabels = ['S', 'I', 'R']
    for i in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        Xi = X[i]
        ax = plt.Subplot(fig, outer[i])
        ax.set_title(r'$N_{pop} = ' + str(N_pops[i]) + ', p_{ER} = ' + str(p_ERs[i]) + '$')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
        for j in range(3):
            ax = plt.Subplot(fig, inner[j])
            percentile_plot(ax, Xi[:,j,:])
            ax.set_ylabel(ylabels[j])
            ax.grid()
            fig.add_subplot(ax)
