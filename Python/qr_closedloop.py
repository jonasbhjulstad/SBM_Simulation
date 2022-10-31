# %%
import numpy as np
import matplotlib.pyplot as plt
from OCP_algorithms import hospital_capacity_objective_solve, quadratic_objective_solve, week_objective_solve

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
qr_model = pf.Polynomial_Model(Nx,Nu,N_output_max,d_max)
qr_features = qr_model.read_csv(DATA_DIR + 'ERR_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/param.csv')


# %%


# %%
qr_model = pf.Polynomial_Model(Nx,Nu,N_output_max,d_max)
qr_features = qr_model.read_csv(DATA_DIR + 'Quantile_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/param.csv')

# %%
from pysindy_casadi_converter import construct_mx_equations
import casadi as cs


# %%
DATA_DIR

# %%
def load_SIR_trajectories():
    trajs = glob.glob(DATA_DIR + 'ERR_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/trajectory*.csv')
    dfs = [pd.read_csv(traj) for traj in trajs[:100] if "Quantile" not in traj]
    
    # print(dfs[0].columns)
    N_traj = len(dfs)
    X = [df[['S', 'I', 'R']].to_numpy() for df in dfs]
    U = [df['p_I'].to_numpy() for df in dfs]

    return X, U

X_sim, U_sim = load_SIR_trajectories()
#get mean of all X_sim
X_mean = np.mean(np.array(X_sim), axis=0)
U_mean = np.mean(np.array(U_sim), axis=0)

# %%
import random
N_sims = 100
p_I0 = 0.1
# #generate integer seeds
# qr_seeds = [random.randint(0, 1000000) for i in range(N_sims)]

# qr_sims = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, p_I0, qr_seeds, Nt, 100)
# qr_sims = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, p_I0, qr_seeds, Nt, 100)
# qr_X_list = [np.array([x[0], x[1], x[2]]) for x in qr_sims]
# qr_X_list = [np.array([x[0], x[1], x[2]]) for x in qr_sims]
graph_seed = 777
G_structure = pf.generate_SIR_ER_graph(N_pop, p_ER, graph_seed)
G = pf.generate_Bernoulli_SIR_Network(G_structure, p_I0, graph_seed, 0)


# %%


# %%
G.reset()
G.initialize()
p_gen = pf.MC_SIR_Params()
p_gen.N_pop = 100
p_gen.p_ER = 1
p_gen.p_I0 = 0.1
qr_param = pf.Regressor_Param()
qr_param.tol = 1e-4
qr_regressor = pf.ERR_Regressor(qr_param)


qr_seeds = [random.randint(0, 1000000) for i in range(N_sims)]

# %%
Nc = 10
Wu = 100
x  = cs.MX.sym('x', Nx)
u  = cs.MX.sym('u', Nu)

p_step = pf.SIR_Param()
p_step.p_R = 0.1

G.reset()
G.initialize()
qr_x_preds = []
qr_us = []
rds = []
qr_true_traj = np.zeros((Nx, (Nc)*7 + 1))
qr_true_traj[:,0] = G.population_count()
qr_true_u = np.zeros(((Nc)*7))
for i in range(Nc):
    rd = pf.MC_SIR_simulations_to_regression(G_structure, p_gen, qr_seeds, Nt)
    rds.append(rd)
    features = qr_regressor.transform_fit(rd, qr_model)
    qr_X_mean = np.mean(np.reshape(rd.X, (N_sims, Nt, Nx)), axis=0)
    qr_U_mean = np.mean(np.reshape(rd.U, (N_sims, Nt, Nu)), axis=0)
    # qr_model.feature_summary(features)
    F_ODE = cs.Function("F_ODE_" + str(i), [x, u], [cs.vertcat(*construct_mx_equations(x, u, qr_model, features))])
    qr_sol, qr_u, qr_x_pred, stats = week_objective_solve(qr_X_mean, qr_U_mean, Wu, F_ODE, Nt, N_pop)
    qr_x_preds.append(qr_x_pred)
    qr_us.append(qr_u)

    p_step.p_I = qr_u[0]
    for j in range(7):
        G.advance(p_step)
        qr_true_traj[:,i*7+j+1] = G.population_count()
        qr_true_u[i*7+j] = p_step.p_I
    # p_gen.p_I_max = 2*np.max(qr_u)



    

# %%
fig, ax = plt.subplots(4)
alphas = np.linspace(0.2, 1, Nc)
t_offset = 0
fake_scale = 1000
for (alpha, rd, qr_x_pred, qr_u) in zip(alphas, rds, qr_x_preds, qr_us):
    t = np.array(range(t_offset, t_offset + Nt))
    t_offset += 7
    plot_sim_comparison_SIR(ax, rd, qr_x_pred, qr_u, t, N_sims, alpha_multiplier=alpha, scale=fake_scale)
t_full = np.array(range(0, qr_true_traj.shape[1]))
control_points = np.array(range(0, qr_true_traj.shape[1], 7))
#scatter at control points
ax[-1].plot(t_full[:-1], qr_true_u, color='black')
_ = [ax[i].scatter(control_points, qr_true_traj[i, control_points]*fake_scale, color='black', marker='x', s=10) for i in range(Nx)]
ax[-1].scatter(control_points[:-1], qr_true_u[control_points[:-1]], color='black', marker='x', s=10)
_ = [x.plot(t_full, xi*fake_scale, color='black') for x, xi in zip(ax[:-1], qr_true_traj)]
ax[-1].set_xlabel('Days')
_ = [x.grid() for x in ax]

# %%
fig, ax = plt.subplots(2)
alphas = np.linspace(0.2, 1, Nc)
t_offset = 0
for (alpha, rd, qr_x_pred, qr_u) in zip(alphas, rds, qr_x_preds, qr_us):
    t = np.array(range(t_offset, t_offset + Nt))
    t_offset += 7
    plot_sim_comparison_SIR(ax, rd, qr_x_pred, qr_u, t, N_sims, alpha_multiplier=alpha, scale=fake_scale, plot_idx=[1])
t_full = np.array(range(0, qr_true_traj.shape[1]))
control_points = np.array(range(0, qr_true_traj.shape[1], 7))
#scatter at control points
ax[-1].plot(t_full[:-1], qr_true_u, color='black')
ax[0].scatter(control_points, qr_true_traj[1, control_points]*fake_scale, color='black', marker='x', s=10)
ax[-1].scatter(control_points[:-1], qr_true_u[control_points[:-1]], color='black', marker='x', s=10)
ax[0].plot(t_full, qr_true_traj[1,:]*fake_scale, color='black')
ax[-1].set_xlabel('Days')
_ = [x.grid() for x in ax]

# %% [markdown]
# 


