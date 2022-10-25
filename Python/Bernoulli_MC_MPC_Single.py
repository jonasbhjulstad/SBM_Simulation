# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.utils import multi_runs
from bokeh.io import output_notebook, show
from ndlib.viz.mpl.TrendComparison import DiffusionTrendComparison
from OCP_algorithms import hospital_capacity_objective_solve, quadratic_objective_solve

from SIR_Plotting_Routines import *
import pysindy as ps
import glob
import pandas as pd


from os.path import basename
import os
import sys


cwd = os.path.dirname(os.path.abspath(''))
# sys.path.append(cwd + '/Cpp/build/Binders/')
import pyFROLS as pf

DATA_DIR = cwd + '/Cpp/data/'
FIGURE_DIR = cwd + '/../../figures/'

# %%
N_pop = 100
p_ER = 1
d_max = 1
N_output_max = 80
Nx = 3
Nu = 1
Nt = 50
er_model = pf.Polynomial_Model(Nx,Nu,N_output_max,d_max)
er_features = er_model.read_csv(DATA_DIR + 'ERR_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/param.csv')
#er_model.feature_summary()


# %%


# %%
qr_model = pf.Polynomial_Model(Nx,Nu,N_output_max,d_max)
qr_features = qr_model.read_csv(DATA_DIR + 'Quantile_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/param.csv')
# qr_model.feature_summary()

# %%
from pysindy_casadi_converter import construct_mx_equations
import casadi as cs


# %%
def load_SIR_trajectories():
    trajs = glob.glob(DATA_DIR + 'ERR_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/trajectory*.csv')
    dfs = [pd.read_csv(traj) for traj in trajs[:100] if "Quantile" not in traj]
    
    print(dfs[0].columns)
    N_traj = len(dfs)
    X = [df[['S', 'I', 'R']].to_numpy() for df in dfs]
    U = [df['p_I'].to_numpy() for df in dfs]

    return X, U

X_sim, U_sim = load_SIR_trajectories()
#get mean of all X_sim
X_mean = np.mean(np.array(X_sim), axis=0)
U_mean = np.mean(np.array(U_sim), axis=0)

# %%
Wu = 100000
t = range(0, Nt)

Nx = X_mean.shape[1]
Nu = 1
xk = cs.MX.sym('X', Nx)
uk = cs.MX.sym('U', Nu)
eqs = cs.vertcat(*construct_mx_equations(xk, uk, qr_model, qr_features))
qr_F = cs.Function('F', [xk, uk], [eqs])

eqs = cs.vertcat(*construct_mx_equations(xk, uk, er_model, er_features))
er_F = cs.Function('F', [xk, uk], [eqs])

(qr_sol, qr_X, qr_U) = quadratic_objective_solve(X_mean, U_mean, Wu, qr_F, Nt)
(er_sol, er_X, er_U) = quadratic_objective_solve(X_mean, U_mean, Wu, er_F, Nt)


# %%

er_sims = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, 100, list(er_U[0]))
qr_sims = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, 100, list(qr_U[0]))
er_X_list = [np.array([x[0], x[1], x[2]]) for x in er_sims]
qr_X_list = [np.array([x[0], x[1], x[2]]) for x in qr_sims]

# %%

fig, ax = plt.subplots(4)

plot_sim_comparison_SIR(ax, er_X_list, X_mean, U_mean, er_X, er_U, t)

# %%
fig, ax = plt.subplots(2)
plot_sim_comparison_I(ax, er_X_list, X_mean, U_mean, er_X, er_U, t)

# %%
er_U

# %%
Wu = 1000
I_max = 25
(qr_sol, qr_X, qr_U) = hospital_capacity_objective_solve(X_mean, U_mean, Wu, I_max, qr_F, Nt)
(er_sol, er_X, er_U) = hospital_capacity_objective_solve(X_mean, U_mean, Wu, I_max, er_F, Nt)

# %%
er_sims_h = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, 1000, list(er_U[0]))
qr_sims_h = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, 1000, list(qr_U[0]))


# %%
er_X_list = [np.array([x[0], x[1], x[2]]) for x in er_sims_h]
qr_X_list = [np.array([x[0], x[1], x[2]]) for x in qr_sims_h]
fig, ax = plt.subplots(4)
plot_sim_comparison_SIR(ax, qr_X_list, X_mean, U_mean, qr_X, qr_U, t)

# %%


# %%
fig, ax = plt.subplots(2)
plot_sim_comparison_I(ax, qr_X_list, X_mean, U_mean, qr_X, qr_U, t)
ax[0].plot(np.ones(Nt)*I_max, color='red', label='Hospital Capacity')
ax[0].set_title('Hospital Capacity under Quantile Regression')

# %%
fig, ax = plt.subplots(2)
plot_sim_comparison_I(ax, er_X_list, X_mean, U_mean, er_X, er_U, t)
ax[0].plot(np.ones(Nt)*I_max, color='red', label='Hospital Capacity')
ax[0].set_title('Hospital Capacity under FROLS Regression')