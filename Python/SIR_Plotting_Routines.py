import numpy as np

def percentile_plot(ax, t, Xi, label = [], color = 'gray'):
    initial = True
    for ci in np.arange(95, 10, -5):
        low = np.percentile(Xi, 50 - ci / 2, axis=0)
        high = np.percentile(Xi, 50 + ci / 2, axis=0)
        # if ci == 85:
        #     ax.plot(t, high, color='k', linestyle='dashed', linewidth=1)
        if(label and initial):
            ax.fill_between(t, low, high, color=color, alpha= .2*(1.0-np.exp(-.01*ci)), label=label)
            initial = False
        else:
            ax.fill_between(t, low, high, color=color, alpha= .2*(1.0-np.exp(-.01*ci)))


def plot_sim_comparison_SIR(ax, X_list, X_mean, U_mean, x, u, t):
    Nx = X_mean.shape[1]
    for i in range(Nx):
        X_sim = np.vstack([x[i,:][:-1] for x in X_list]).T
        percentile_plot(ax[i], t, X_sim.T, label='Controlled')
        ax[i].plot(t, x[i, :-1].T, color='k', linestyle='dotted', label='MPC')
        ax[i].plot(t, X_mean[:,i].T, color='k', linestyle='dashed', label='Uncontrolled mean')
    ax[-1].plot(t, U_mean, color='k', linestyle='dashed', label='Uncontrolled')
    
    if (u.shape[0] < t.shape[0]):
        ax[-1].plot(t[::7][:u.shape[0]], u[:], color='k', linestyle='dotted', label='MPC controlled')
    else:
        ax[-1].plot(t, u.T, color='k', linestyle='dotted', label='MPC controlled')
    _ = [x.set_ylabel(s) for x, s in zip(ax, ['S', 'I', 'R', r'$p_I$'])]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.grid() for x in ax]
    ax[0].legend(loc=1, prop={'size': 6})
    ax[-1].legend(loc=1, prop={'size': 6})
    ax[-1].set_yscale('log')

def plot_sim_comparison_I(ax, X_list, X_mean, U_mean, x, u, t):
    X_sim = np.vstack([x[1,:][:-1] for x in X_list]).T
    percentile_plot(ax[0], t, X_sim.T, label='Controlled')
    ax[0].plot(t[:(x.shape[1]-1)], x[1, :-1].T, color='k', linestyle='dotted', label='MPC Prediction')
    # ax[0].plot(t, X_mean[:,1].T, color='r', linestyle='dashed', label='Uncontrolled mean')
    # ax[-1].plot(t, U_mean, color='r', linestyle='dashed', label='Uncontrolled mean')
    ax[-1].plot(t, u.T[:t.shape[0]], color='k', linestyle='dotted', label='MPC controlled')
    _ = [x.set_ylabel(s) for x, s in zip(ax, ['I', r'$p_I$'])]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.grid() for x in ax]
    ax[0].legend(loc=1, prop={'size': 6})
    ax[-1].legend(loc=1, prop={'size': 6})

def plot_sim_comparison(ax, X_sim_list, X_sol_list, U_mean, u, t):
    X_sim = np.vstack([x[1,:][:-1] for x in X_sol_list]).T
    percentile_plot(ax[0], t, X_sim.T, 'Controlled MC-simulations')
    percentile_plot(ax[0], t, X_sim_list.T, 'Uncontrolled MC-simulations')
    ax[-1].plot(t, U_mean, color='k', linestyle='dashed', label='Uncontrolled mean')
    ax[-1].plot(t, u.T, color='k', linestyle='dotted', label='MPC')
    _ = [x.set_ylabel(s) for x, s in zip(ax, ['Infected', r'Interaction Probability'])]
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.grid() for x in ax]

def fake_plot(ax, t, X_stack, U_stack, X_mean, er_X_list, U_mean, er_X, er_U):
    fake_scaling = 1000
    percentile_plot(ax[0], t[:X_stack.shape[1]], X_stack*fake_scaling, label="Uncontrolled", color='red')
    percentile_plot(ax[1], t[:U_stack.shape[1]], U_stack, label="Uncontrolled", color='red')
    scaled_er_X_list = [x*fake_scaling for x in er_X_list]
    plot_sim_comparison_I(ax, scaled_er_X_list, X_mean*fake_scaling, U_mean, er_X*fake_scaling, er_U, t)
    ax[0].set_ylabel('Number of Infected')
    #Remove y-axis ticks
    # ax[0].set_yticks([])
    # ax[-1].set_yticks([])
    ax[-1].set_yticklabels([])
    ax[-1].set_xticklabels([])
    ax[-1].set_ylabel('Contact Probability')
