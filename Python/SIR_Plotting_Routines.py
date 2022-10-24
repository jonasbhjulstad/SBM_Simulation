import numpy as np

def percentile_plot(ax, t, Xi, label = []):
    initial = True
    for ci in np.arange(95, 10, -5):
        low = np.percentile(Xi, 50 - ci / 2, axis=0)
        high = np.percentile(Xi, 50 + ci / 2, axis=0)
        # if ci == 85:
        #     ax.plot(t, high, color='k', linestyle='dashed', linewidth=1)
        if(label and initial):
            ax.fill_between(t, low, high, color='gray', alpha= .3*(1.0-np.exp(-.01*ci)), label=label)
            initial = False
        else:
            ax.fill_between(t, low, high, color='gray', alpha= .3*(1.0-np.exp(-.01*ci)))


def plot_sim_comparison_SIR(ax, X_list, X_mean, U_mean, x, u, t):
    Nx = X_mean.shape[1]
    for i in range(Nx):
        X_sim = np.vstack([x[i,:][:-1] for x in X_list]).T
        percentile_plot(ax[i], t, X_sim.T)
        ax[i].plot(t, x[i, :-1].T, color='k', linestyle='dotted')
        ax[i].plot(t, X_mean[:,i].T, color='k', linestyle='dashed')
    ax[-1].plot(t, U_mean, color='k', linestyle='dashed')
    ax[-1].plot(t, u.T, color='k', linestyle='dotted')
    _ = [x.set_ylabel(s) for x, s in zip(ax, ['S', 'I', 'R', r'$p_I$'])]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.grid() for x in ax]

def plot_sim_comparison_I(ax, X_list, X_mean, U_mean, x, u, t):
    X_sim = np.vstack([x[1,:][:-1] for x in X_list]).T
    percentile_plot(ax[0], t, X_sim.T)
    ax[0].plot(t, x[1, :-1].T, color='k', linestyle='dotted')
    ax[0].plot(t, X_mean[:,1].T, color='k', linestyle='dashed')
    ax[-1].plot(t, U_mean, color='k', linestyle='dashed')
    ax[-1].plot(t, u.T, color='k', linestyle='dotted')
    _ = [x.set_ylabel(s) for x, s in zip(ax, ['I', r'$p_I$'])]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.grid() for x in ax]

def plot_sim_comparison(ax, X_sim_list, X_sol_list, U_mean, u, t):
    X_sim = np.vstack([x[1,:][:-1] for x in X_sol_list]).T
    percentile_plot(ax[0], t, X_sim.T)
    percentile_plot(ax[0], t, X_sim_list.T)
    ax[-1].plot(t, U_mean, color='k', linestyle='dashed')
    ax[-1].plot(t, u.T, color='k', linestyle='dotted')
    _ = [x.set_ylabel(s) for x, s in zip(ax, ['Infected', r'Interaction Probability'])]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.grid() for x in ax]