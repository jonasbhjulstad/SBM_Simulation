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
cwd = '/home/man/Documents/Bernoulli_MC/Python'

DATA_DIR = cwd + '/../Cpp/data/'
FIGURE_DIR = cwd + '/../../figures/'

# %%
d_max = 1
N_output_max = 80
Nx = 3
Nu = 1
Nt = 56
er_model = pf.Polynomial_Model(Nx,Nu,d_max, N_output_max)
# er_features = er_model.read_csv(DATA_DIR + 'ERR_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/param.csv')


# %%
qr_model = pf.Polynomial_Model(Nx,Nu,d_max, N_output_max)
# qr_features = qr_model.read_csv(DATA_DIR + 'Quantile_Simulation_SIR_' + str(N_pop) + '_' + str(p_ER) + '/param.csv')

# %%
from pysindy_casadi_converter import construct_mx_equations
import casadi as cs


# %%
import random
N_sims = 50
p_I0 = 0.2
x_names = [r'$S_{t}$', r'$I_{t}$', r'$R_{t}$']
u_names = [r'$p_{I}$']
y_names = [r'$S_{t+1}$', r'$I_{t+1}$', r'$R_{t+1}$']
# #generate integer seeds
# qr_seeds = [random.randint(0, 1000000) for i in range(N_sims)]
from copy import deepcopy
# er_sims = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, p_I0, er_seeds, Nt, 100)
# qr_sims = pf.Bernoulli_SIR_MC_Simulations(N_pop, p_ER, p_I0, qr_seeds, Nt, 100)
# er_X_list = [np.array([x[0], x[1], x[2]]) for x in er_sims]
# qr_X_list = [np.array([x[0], x[1], x[2]]) for x in qr_sims]
graph_seed = 642
p_gen = pf.MC_SIR_Params()
p_gen.N_pop = 20
p_gen.p_I0 = 0.2
er_param = pf.Regressor_Param()
er_param.tol = 1e-6
er_param.theta_tol = 1e-6
er_param.N_terms_max = 2
er_regressor = pf.ERR_Regressor(er_param)
qr_param = pf.Quantile_Param()
qr_param.tol = 1e-12
qr_param.N_terms_max = 2
qr_param.theta_tol = 1e-16
tau = .7
qr_param.tau = tau
qr_regressor = pf.Quantile_Regressor(qr_param)

qr_param.tau = 1-tau
reverse_qr_regressor = pf.Quantile_Regressor(qr_param)
from copy import deepcopy
er_seeds = [random.randint(0, 1000000) for i in range(N_sims)]
def uncontrolled_trajectory_generation(N_pop, p_ER):
    p_gen = pf.MC_SIR_Params()
    p_gen.R0_max = 1.5
    p_gen.N_pop = N_pop
    p_gen.p_ER = p_ER
    p_gen.p_I0 = 0.2
    p_gen.N_I_min = 0
    #set numpy random seed
    np.random.seed(seed)
    #generate integer seeds
    seeds = [random.randint(0, 1000000) for i in range(N_sims)]

    G_structure = pf.generate_SIR_ER_graph(N_pop, p_ER, seed)
    G = pf.generate_Bernoulli_SIR_Network(G_structure, p_I0, seed, 0)
    G_mpc = G
    #generate integer seeds
    seeds = [random.randint(0, 1000000) for i in range(N_sims)]
    rd = pf.MC_SIR_simulations_to_regression(G_structure, p_gen, seeds, Nt)
    X_sim = np.reshape(rd.X, (N_sims, Nt, Nx))

    U_sim = np.reshape(rd.U, (N_sims, Nt, Nu))
    X_mean = np.mean(X_sim, axis=0)
    U_mean = np.mean(U_sim, axis=0)
    max_infected_idx = np.argmax(np.max(X_sim[:,:,1], axis=1))
    X0 = X_sim[max_infected_idx,:,:]
    U0 = U_sim[max_infected_idx,:,:]
    uncontrolled_data = {'X_mean': X_mean, 'U_mean': U_mean, 'Gs': G_structure, 'X0': X0, 'U0': U0, 'N_pop': N_pop, 'p_ER': p_ER}
    return uncontrolled_data, rd


def openloop_solve(uc_data, rd, regressor, model, N_sims, Nt, Wu, file_prefix = ''):

    # feature_S = regressor.transform_fit(rd.X, rd.U, rd.Y[:,0], model)
    # feature_I = regressor.transform_fit(rd.X, rd.U, rd.Y[:,1], model)
    # feature_R = regressor.transform_fit(rd.X, rd.U, rd.Y[:,2], model)
    features = regressor.transform_fit(rd.X, rd.U, rd.Y, model)

    # features = [feature_S, feature_I, feature_R]
    model.feature_summary(features)
    model.write_latex(features, DATA_DIR + '/latex/param/' + file_prefix + 'param_{}_{}.tex'.format(N_pop, p_ER), x_names, u_names, y_names, False, "&")

    x = cs.MX.sym('x', 3)
    u = cs.MX.sym('u', 1)

    xdot = cs.vertcat(*construct_mx_equations(x, u, er_model, features))
    #Quantile Regression and optimiation
    F_ODE = cs.Function("F_ODE", [x, u], [xdot])
    log_filename = DATA_DIR + '/latex/log/' + file_prefix + 'week_objective_solve_{}_{}.txt'.format(N_pop, p_ER)    
    sol, u_sol, x_pred, stats = week_objective_solve(uc_data['X0'], uc_data['U0'], Wu, F_ODE, Nt, N_pop, log_file=log_filename)
    x_pred = model.simulate(uc_data['X0'][0,:], u_sol, Nt, features)

    #MPC simulations
    mpc_sir_p = [pf.SIR_Param() for i in range(Nt)]
    for i in range(Nt):
        mpc_sir_p[i].p_R = 0.1
        mpc_sir_p[i].p_I = u_sol[i]
        mpc_sir_p[i].N_I_min = 0        

    seeds = [random.randint(0, 1000000) for i in range(N_sims)]
    mpc_rd = pf.MC_SIR_simulations_to_regression(uc_data['Gs'], p_gen, mpc_sir_p, seeds, Nt)


    openloop_data = {'X': mpc_rd.X, 'U': mpc_rd.U, 'features': features,
               'x_pred': x_pred, 'u_sol': u_sol, 'sol': sol, 'stats': stats}

    return openloop_data

def openloop_solve_from_csv(rd, param_filename, model, N_sims, Nt, Wu, N_pop, p_ER, file_prefix = ''):

    # feature_S = regressor.transform_fit(rd.X, rd.U, rd.Y[:,0], model)
    # feature_I = regressor.transform_fit(rd.X, rd.U, rd.Y[:,1], model)
    # feature_R = regressor.transform_fit(rd.X, rd.U, rd.Y[:,2], model)
    # features = [feature_S, feature_I, feature_R]


    features = model.read_csv(param_filename)

    model.write_latex(features, DATA_DIR + '/latex/param/' + file_prefix + 'param_{}_{}.tex'.format(N_pop, p_ER), x_names, u_names, y_names, False, "&")

    x = cs.MX.sym('x', 3)
    u = cs.MX.sym('u', 1)

    xdot = cs.vertcat(*construct_mx_equations(x, u, er_model, features))
    #Quantile Regression and optimiation
    F_ODE = cs.Function("F_ODE", [x, u], [xdot])
    log_filename = DATA_DIR + '/latex/log/' + file_prefix + 'week_objective_solve_{}_{}.txt'.format(N_pop, p_ER)    
    sol, u_sol, x_pred, stats = week_objective_solve(rd.X[0], rd.U[0], Wu, F_ODE, Nt, N_pop, log_file=log_filename)
    # x_pred = model.simulate(rd.X[0][0,:], u_sol, Nt, features)
    #MPC simulations

    mpc_sir_p = [pf.SIR_Param() for i in range(Nt)]
    for i in range(Nt):
        mpc_sir_p[i].p_R = 0.1
        mpc_sir_p[i].p_I = u_sol[i]
        mpc_sir_p[i].N_I_min = 0        

    seeds = [random.randint(0, 1000000) for i in range(N_sims)]
    mpc_rd = pf.MC_SIR_simulations_to_regression(uc_data['Gs'], p_gen, mpc_sir_p, seeds, Nt)


    openloop_data = {'X': mpc_rd.X, 'U': mpc_rd.U, 'features': features,
               'x_pred': x_pred, 'u_sol': u_sol, 'sol': sol, 'stats': stats}

    return openloop_data
    

# %%
seed = random.randint(0, 1000000)
Wu = 50
uc_datas = []
G_param_pairs = []
N_pops = reversed([10, 50, 100])
p_ERs = [0.1, 0.5, 1.0]
t = np.array(range(Nt))
uncontrolled_traj_fname = lambda p: DATA_DIR + '/latex/Figures/MPC_Trajectory_comparison_{}_{}.pdf'.format(p[0], p[1])
for N_pop in N_pops:
    for p_ER in p_ERs:
        G_param_pairs.append((N_pop, p_ER))
uc_rds = []

for p in G_param_pairs:
    print('Solving for N_pop = {}, p_ER = {}'.format(p[0], p[1]))
    uc_data, rd = uncontrolled_trajectory_generation(p[0], p[1])

    #convert p[1] to a string
    p_str = str(p[1])
    #remove the decimal point if equal to 1
    if p_str[-1] == '0':
        p_str = p_str[:1]
    uc_filename = DATA_DIR + '/Bernoulli_SIR_MC_{}_{}/regression_data.csv'.format(p[0], p_str)
    er_param_filename = DATA_DIR + '/ERR_Simulation_SIR_{}_{}/param.csv'.format(p[0], p_str)
    qr_param_filename = DATA_DIR + '/Quantile_Simulation_SIR_{}_{}/param.csv'.format(p[0], p_str)
    er_data = openloop_solve_from_csv(rd, er_param_filename, er_model, N_sims, Nt, Wu, N_pop, p_ER, file_prefix='er_')
    qr_data = openloop_solve_from_csv(rd, qr_param_filename, qr_model, N_sims, Nt, Wu, N_pop, p_ER, file_prefix='qr_')
    # er_data = openloop_solve(uc_data, rd, er_regressor, er_model, N_sims, Nt, Wu, file_prefix='er_')
    # qr_model.feature_susmmary(er_data['features'])
    # qr_data = openloop_solve(uc_data, rd, qr_regressor, qr_model, N_sims, Nt, Wu, file_prefix='qr_')
    # qr_model.feature_summary(qr_data['features'])
    #plot er_data['x_pred']
    # _ = [x.plot(t, er_data['x_pred'][i,:t.shape[0]].T, label='ER') for i, x in enumerate(ax)]
    mpc_trajectory_plot(p, er_data, qr_data, t, uncontrolled_traj_fname(p))

    uc_datas.append(uc_data)
    uc_rds.append(rd)

# %%

t = np.array(range(Nt))
uncontrolled_traj_fname = DATA_DIR + '/latex/Figures/MPC_Trajectory_comparison.pdf'
plot_uncontrolled(G_param_pairs, uc_rds, t, uncontrolled_traj_fname)

# %%


