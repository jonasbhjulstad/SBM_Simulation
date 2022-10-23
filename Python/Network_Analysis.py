#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.utils import multi_runs
# from aesara import tensor as at
from bokeh.io import output_notebook, show
# from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.TrendComparison import DiffusionTrendComparison
import pysindy as ps
# from pydmd import DMD
# from ParameterConfig import DATA_DIR, FIGURE_DIR, ROOT_DIR
import glob
import pandas as pd
from os.path import basename
import os
import sys


cwd = os.path.dirname(os.path.abspath(''))
print(cwd)
# sys.path.append(cwd + '/Cpp/build/Binders/')
# import pyFROLS as pf

DATA_DIR = '/home/man/Documents/Bernoulli_MC/Cpp/data/Bernoulli_SIR_MC_100_1/'
FIGURE_DIR = cwd + '/../../figures/'
def load_SIR_trajectories():
    trajs = glob.glob(DATA_DIR + '*.csv')
    dfs = [pd.read_csv(traj) for traj in trajs[:100] if "Quantile" not in traj]
    N_traj = len(dfs)
    X = [df[['S', 'I', 'R']].to_numpy() for df in dfs]
    U = [df['p_I'].to_numpy() for df in dfs]

    return X, U

def plot_separate(X, reg_model):
    x_grouped = [X[:,i,:] for i in range(3)]
    X_list = [x.T for x in X]
    fig, ax = plt.subplots(3,2)
    t = np.linspace(0,1,x_grouped[0].shape[1])
    sim = reg_model.simulate(x0=X_list[0][0,:], t=np.linspace(0,1, t.shape[0]))

    for ci in np.arange(95, 10, -5):
        for (i, x) in enumerate(x_grouped):
            low = np.percentile(x, 50 - ci / 2, axis=2)
            high = np.percentile(x, 50 + ci / 2, axis=2)
            ax[i,0].fill_between(t, low, high, color='gray', alpha= np.exp(-.01*ci))

    ax[0,0].set_title("Susceptible")
    ax[1,0].set_title("Infected")
    ax[2,0].set_title("Recovered")
    _ = [x.grid() for x in ax[:,0]]
    _ = [x.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) for x in ax[:,0]]
    _ = [x.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False) for x in ax[:,0]]
    
    for i in range(N_sim):
        x0 = X_list[i][0,:]
        sim = reg_model.simulate(x0=x0, t=np.linspace(0,1, t.shape[0]))
        ax[0,1].plot(t,sim[:,0], color='k')
        ax[1,1].plot(t, sim[:,1], color='k')
        ax[2,1].plot(t, sim[:,2], color='k')

    ax[0,1].set_title("Susceptible")
    ax[1,1].set_title("Infected")
    ax[2,1].set_title("Recovered")
    _ = [x.grid() for x in ax[:,1]]
    _ = [x.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) for x in ax[:,1]]
    _ = [x.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False) for x in ax[:,1]]
    
def plot_merged(X, U, reg_model):
    x_grouped = [np.concatenate([x[:,i][:, np.newaxis] for x in X], axis=1) for i in range(3)]
    X_list = [x.T for x in X]
    fig, ax = plt.subplots(3)
    t = np.linspace(0,1,X[0].shape[0])
    sim = reg_model.simulate(x0=X[0][0,:], u=U, t=np.linspace(0,1, t.shape[0]))

    for ci in np.arange(95, 10, -5):
        for (i, x) in enumerate(x_grouped):
            low = np.percentile(x, 50 - ci / 2, axis=1)
            high = np.percentile(x, 50 + ci / 2, axis=1)
            ax[i].fill_between(t, low, high, color='gray', alpha= np.exp(-.01*ci))

    ax[0].set_title("Susceptible")
    ax[1].set_title("Infected")
    ax[2].set_title("Recovered")
    _ = [x.grid() for x in ax[:]]
    _ = [x.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) for x in ax]
    _ = [x.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False) for x in ax]
    
    X0 = X_list[0][:,0]
    sim = reg_model.simulate(x0=X0, t=np.linspace(0,1, t.shape[0]), u=U)

    ax[0].plot(t[0], X0[0],color='k')
    ax[1].plot(t[0], X0[1], color='k')
    ax[2].plot(t[0], X0[2], color='k')

    ax[0].plot(t[1:], sim[:, 0],color='k')
    ax[1].plot(t[1:], sim[:, 1], color='k')
    ax[2].plot(t[1:], sim[:, 2], color='k')

    return fig, ax


# In[2]:
if __name__ == '__main__':

    X, U = load_SIR_trajectories()

    U_mean = np.mean(U, axis=0)

    Nt = X[0].shape[0]
    N_sim = len(X)
    # In[8]:


    t = [*list(range(Nt))]*N_sim

    # In[10]:


    # from Quantile_STLSQ import Quantile_STLSQ
    # from Quantile_FROLS import Quantile_FROLS
    # reg_model = ps.SINDy(Quantile_STLSQ(tau=.95, threshold=1e-6, alpha=1e-6))
    # reg_model = ps.SINDy(Quantile_FROLS(tau=.95, verbose=True, max_iter = 3))
    lowPolyLib = ps.PolynomialLibrary(degree=2)
    # reg_model = ps.SINDy(ps.STLSQ(threshold=.5, alpha=10), feature_library=lowPolyLib)
    frols = ps.FROLS()
    reg_model = ps.SINDy(frols, feature_library=lowPolyLib)

    reg_model.fit(X,u=U, t=np.linspace(0,1,Nt), multiple_trajectories=True)
    reg_model.print()

    fig, ax = plot_merged(X, U_mean, reg_model)

    fig.savefig(CPP_Figure_Dir + 'SIR_merged.png', bbox_inches='tight')
    fig.show()

