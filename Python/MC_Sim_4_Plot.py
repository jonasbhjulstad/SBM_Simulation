import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from ParameterConfig import DATA_DIR, FIGURE_DIR, ROOT_DIR

def sim_fnames(N_pop, p_ER):
    return DATA_DIR + 'SIR_trajectories_' + str(N_pop) + '_' + str(p_ER) + '.csv'
def load_SIR(filename):
    X = np.loadtxt(filename, delimiter=',')
    return X.reshape(X.shape[0], 3, int(X.shape[1]/3))
N_pops = [20, 50, 20, 50]
p_ERs = [.1, .1, 1.0, 1.0]

fnames = [sim_fnames(N_pop, p_ER) for N_pop, p_ER in zip(N_pops, p_ERs)]
#load data

X = [load_SIR(fname) for fname in fnames]


fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

def percentile_plot(ax, Xi):
    for ci in np.arange(95, 10, -5):
        low = np.percentile(Xi, 50 - ci / 2, axis=0)
        high = np.percentile(Xi, 50 + ci / 2, axis=0)
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

fig.savefig(FIGURE_DIR + 'MC_Simulation_4_Plot.svg', format='svg')