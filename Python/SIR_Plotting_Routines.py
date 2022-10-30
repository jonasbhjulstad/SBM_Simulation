import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import axes_grid
import matplotlib.gridspec as gridspec
def percentile_plot(ax, t, Xi, label = [], color = 'gray', alpha_mul=1., hatch=''):
    initial = True
    lows = []
    highs = []
    for ci in np.arange(45, 0, -5):
        # lows.append(np.percentile(Xi, 50 - ci / 2, axis=0))
        low = np.percentile(Xi, 50 - ci, axis=0)
        # highs.append(np.percentile(Xi, 50 + ci / 2, axis=0))
        high = np.percentile(Xi, 50 + ci, axis=0)
        if ci == 5:
            ax.fill_between(t, low, high, color=color, alpha=alpha_mul, label=label, hatch=hatch)
        else:
            ax.fill_between(t, low, high, color=color, alpha=alpha_mul, hatch=hatch)
            


    # for i in range(len(lows)-1):
    #     ax.fill_between(t, lows[i], lows[i+1], color=color, alpha= alpha_mul*(1.0-np.exp(-.01*ci)), hatch=hatch)
    # for i in reversed(range((len(highs)-1))):
    #     ax.fill_between(t, highs[i], highs[i+1], color=color, alpha= alpha_mul*(1.0-np.exp(-.01*ci)), hatch=hatch)
    # ax.fill_between(t, lows[-1], highs[0], color=color, alpha= alpha_mul*(1.0-np.exp(-.01*ci)), label=label, hatch=hatch)
    

def single_percentile_plot(ax, t, Xi, tau, label = [], color = 'gray', alpha=1., linestyle='-'):
    perc = np.percentile(Xi, tau, axis=0)
    ax.plot(t, perc, color=color, alpha=alpha, label=label, linestyle=linestyle)

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

def mpc_single_plot(ax, data, t, label=None):
    Nx = data['X'].shape[1]
    Nt = t.shape[0]
    N_sims = int(data['X'].shape[0]/Nt)

    X_sim = np.reshape(data['X'], (N_sims, Nt, Nx))
    _ = [percentile_plot(ax[j], t, X_sim[:,:,j], color='gray', hatch='', alpha_mul=0.2, label='MC-simulations') for j in range(Nx)]
    single_percentile_plot(ax[0], t, X_sim[:,:,0], 5, color='black',linestyle='dotted', alpha=1, label='95-percentile')
    single_percentile_plot(ax[1], t, X_sim[:,:,1], 95, color='black',linestyle='dotted', alpha=1, label='95-percentile')
    single_percentile_plot(ax[2], t, X_sim[:,:,2], 95, color='black',linestyle='dotted', alpha=1, label='95-percentile')
    _ = [ax[j].plot(t, data['x_pred'][j,:-1].T, color='black', linestyle='--', label='Prediction') for j in range(Nx)]
def mpc_plot(er_ax, qr_ax, u_ax, data_er, data_qr, t):
    
    mpc_single_plot(er_ax, data_er, t, label='FROLS')
    labels = mpc_single_plot(qr_ax, data_qr, t)
    u_ax.plot(t, data_er['u_sol'], color='black', linestyle='--', label='FROLS_Regression')
    u_ax.plot(t, data_qr['u_sol'], color='black', linestyle='dashdot', label='Quantile Regression')

def mpc_trajectory_plot(Gp, er_d, qr_d, t, filename):
    Nx = er_d['X'].shape[1]
    Nt = t.shape[0]
    N_sims = er_d['X'].shape[0]/Nt

    fig = plt.figure()
    gs0 = gridspec.GridSpec(2, 2, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[0,0])
    gs01 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[0,1])
    gs1 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[1,:])

    ax00 = [fig.add_subplot(gs00[i,0]) for i in range(Nx)]
    ax01 = [fig.add_subplot(gs01[i,0]) for i in range(Nx)]
    ax1 = fig.add_subplot(gs1[0])
    axs = [ax00, ax01]
    custom_lines = [Line2D([0], [0], color='k', linestyle='--'),
                Line2D([0], [0], color='k', linestyle='dashdot')]
    custom_labels = ['FROLS MPC', 'Quantile MPC']
    if (er_d['stats']['success']) and (qr_d['stats']['success']):
        _ = [[x.grid('small') for x in ax] for ax in axs]
        ax1.grid('small')
        mpc_plot(ax00, ax01, ax1, er_d, qr_d, t)
        ax00[0].set_title('FROLS Regression')
        ax01[0].set_title('Quantile Regression')
        _ = [x.set_xticklabels([]) for x in ax00[:-1]]
        _ = [x.set_ylabel(s) for x, s in zip(ax00, [r'$N_S$', r'$N_I$', r'$N_R$'])]
        # _ = [x.set_ylabel(s) for x, s in zip(ax10, [r'$N_S$', r'$N_I$', r'$N_R$'])]
        _ = [x.set_xticklabels([]) for x in ax01[:-1]]
        _ = ax1.set_xlabel(r'$t$')
        _ = ax1.set_ylabel(r'$p_I$')
        for i in range(Nx):
            y_lim = min(max(ax00[i].get_ylim()[1], ax01[i].get_ylim()[1]), Gp[0])
            ax00[i].set_ylim(0, y_lim)
            ax01[i].set_ylim(0, y_lim)
        ax01[-1].legend(loc='upper left', prop={'size': 4})
        _ = [[x.set_xlim(t[0], t[-1]) for x in ax] for ax in axs]
        ax1.set_xlim(t[0], t[-1])
        ax1.legend(custom_lines, custom_labels, loc=1, prop={'size': 6})

        fig.savefig(filename, bbox_inches='tight')
        _ = [[x.clear() for x in ax] for ax in axs]
        ax1.clear()

def plot_uncontrolled(G_param_pairs, uc_datas, t, filename):
    Nx = uc_datas[0]['X'].shape[1]
    Nt = t.shape[0]
    N_sims = int(uc_datas[0]['X'].shape[0]/Nt)

    fig = plt.figure()
    gs0 = gridspec.GridSpec(2, 2, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[0,0])
    gs01 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[0,1])
    gs10 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[1,0])
    gs11 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs0[1,1])

    ax00 = [fig.add_subplot(gs00[i,0]) for i in range(Nx)]
    ax01 = [fig.add_subplot(gs01[i,0]) for i in range(Nx)]
    ax10 = [fig.add_subplot(gs10[i,0]) for i in range(Nx)]
    ax11 = [fig.add_subplot(gs11[i,0]) for i in range(Nx)]
    axs = [ax00, ax01, ax10, ax11]
    t = np.array(range(Nt))
    # qr_X_sim = np.reshape(qr_rd.X, (N_sims, Nt, Nx))
    # qr_mpc_X_sim = np.reshape(qr_mpc_rd.X, (N_sims, Nt, Nx))

    # er_X_sim = np.reshape(er_rd.X, (N_sims, Nt, Nx))
    # er_mpc_X_sim = np.reshape(er_mpc_rd.X, (N_sims, Nt, Nx))
    for i, gp in enumerate(G_param_pairs):
        ax = axs[i]
        X_sim = np.reshape(uc_datas[i]['X'], (N_sims, Nt, Nx))
        _ = [percentile_plot(ax[j], t, X_sim[:,:,j], color='gray', hatch='', alpha_mul=0.2) for j in range(Nx)]
        single_percentile_plot(ax[0], t, X_sim[:,:,0], 5, color='black',linestyle='dotted', alpha=1)
        single_percentile_plot(ax[1], t, X_sim[:,:,1], 95, color='black',linestyle='dotted', alpha=1)
        single_percentile_plot(ax[2], t, X_sim[:,:,2], 95, color='black',linestyle='dotted', alpha=1)
        ax[0].set_title(r'$N_{{pop}} = {}, p_{{ER}} = {}$'.format(gp[0], gp[1]))
    _ = [x.set_xticklabels([]) for x in ax00]
    _ = [x.set_ylabel(s) for x, s in zip(ax00, [r'$N_S$', r'$N_I$', r'$N_R$'])]
    _ = [x.set_ylabel(s) for x, s in zip(ax10, [r'$N_S$', r'$N_I$', r'$N_R$'])]
    _ = [x.set_xticklabels([]) for x in ax01]
    _ = [[x.grid() for x in ax] for ax in axs]
    _ = [[x.set_xlim(t[0], t[-1]) for x in ax] for ax in axs]

    _ = ax10[-1].set_xlabel(r'$t$')
    _ = ax11[-1].set_xlabel(r'$t$')

    fig.savefig(filename, bbox_inches='tight')