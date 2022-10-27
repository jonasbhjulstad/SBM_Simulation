import numpy as np

def percentile_plot(ax, t, Xi, label = [], color = 'gray', alpha_mul=1., hatch=''):
    initial = True
    for ci in np.arange(95, 10, -5):
        low = np.percentile(Xi, 50 - ci / 2, axis=0)
        high = np.percentile(Xi, 50 + ci / 2, axis=0)
        if (ci == 95) and label:
            ax.fill_between(t, low, high, color=color, alpha= alpha_mul*.2*(1.0-np.exp(-.01*ci)), label=label, hatch=hatch)
        else:
            ax.fill_between(t, low, high, color=color, alpha= alpha_mul*.2*(1.0-np.exp(-.01*ci)), hatch=hatch)
def plot_sim_comparison_SIR(ax, rd, x, u, t, N_sims, alpha_multiplier=1, scale=1, plot_idx=[0,1,2], hatch='', simple=False, plot_u = False):
    Nx = 3
    Nu = 1
    Nt = t.shape[0]
    X_sim = np.reshape(rd.X, (N_sims, Nt, Nx))*scale
    X_mean = np.mean(X_sim, axis=0)
    U_mean = np.mean(np.reshape(rd.U, (N_sims, Nt, Nu)), axis=0)
    for i, idx in enumerate(plot_idx):
        percentile_plot(ax[i], t, X_sim[:,:,idx], label='Uncontrolled', alpha_mul=alpha_multiplier, hatch=hatch)
        # ax[i].plot(t, x[i, :-1].T, color='k', linestyle='dotted', label='MPC', alpha=alpha_multiplier)
        # ax[i].plot(t, X_mean[:,i].T, color='k', linestyle='dashed', label='Uncontrolled mean', alpha=alpha_multiplier)
    # ax[-1].plot(t, U_mean, color='k', linestyle='dashed', label='Uncontrolled', alpha=alpha_multiplier)
    
    if (u.shape[0] < t.shape[0]):
        ax[-1].plot(t[::7][:u.shape[0]], u, color='k', linestyle='dotted', label='MPC controlled', alpha=alpha_multiplier)
    else:
        ax[-1].plot(t, u, color='k', linestyle='--', label='MPC controlled', alpha=alpha_multiplier)
    if(simple):
        labels = ['Susceptible', 'Infected', 'Recovered', r'Contact probability']
    else:
        labels = [r'$N_S$', r'$N_I$', r'$N_R$', r'$p_I$']
    for i, idx in enumerate(plot_idx):
        ax[i].set_ylabel(labels[idx])
        ax[i].set_xticklabels([])
        ax[i].grid()
    ax[-1].set_xlabel('t')
    # ax[0].legend(loc=1, prop={'size': 6})
    # ax[-1].legend(loc=1, prop={'size': 6})
    ax[-1].set_yscale('log')
    ax[-1].set_ylabel(labels[-1])

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
