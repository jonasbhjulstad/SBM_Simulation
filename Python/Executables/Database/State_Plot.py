from Database.Simulation_Tables import *
import numpy as np
def plot_SIR_percentile_trajectory(p_out_id, graph_id, ax, engine):
    params = read_sim_param(p_out_id, graph_id, engine)

    df = read_total_graph_state(p_out_id, graph_id, engine)
    #average S, I, R over simulations, drop simulation index
    df_mean = df.groupby('t').mean().reset_index()
    df_mean.drop('simulation', axis=1, inplace=True)

    # df_mean = df.groupby(['t', 'simulation']).agg({'s': 'mean', 'i': 'mean', 'r': 'mean'}).reset_index()
    # std = np.std(trajectories, axis=0)
    #get percentiles
    N_perc = 11
    mid_idx = int(np.floor(N_perc/2))
    percentiles = np.linspace(.05, .95, N_perc)
    df_perc = df.groupby('t').quantile(percentiles)
    df_perc.drop('simulation', axis=1, inplace=True)
    #transform t from index to column
    df_perc.reset_index(inplace=True, level=0)
    #transform index to column 'q'
    df_perc['q'] = df_perc.index
    # percentiles = [np.percentile(trajectories, p, axis=0) for p in np.linspace(5, 95, 11)]
    Nt = params['Nt']


    for j in range(mid_idx):
        df_hi = df_perc[df_perc['q'] == percentiles[mid_idx + j]][['s', 'i', 'r']]
        df_lo = df_perc[df_perc['q'] == percentiles[mid_idx-j]][['s', 'i', 'r']]
        ax[0].fill_between(np.arange(Nt+1), df_lo['s'], df_hi['s'], color='gray', alpha=0.1)
        ax[1].fill_between(np.arange(Nt+1), df_lo['i'], df_hi['i'], color='gray', alpha=0.1)
        ax[2].fill_between(np.arange(Nt+1), df_lo['r'], df_hi['r'], color='gray', alpha=0.1)
    ax[0].plot(np.arange(Nt+1), df_mean['s'], color='k', linewidth=2, linestyle='--')
    ax[1].plot(np.arange(Nt+1), df_mean['i'], color='k', linewidth=2, linestyle='--')
    ax[2].plot(np.arange(Nt+1), df_mean['r'], color='k', linewidth=2, linestyle='--')