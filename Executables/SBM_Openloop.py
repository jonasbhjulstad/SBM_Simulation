from casadi import *
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
Project_root = "/home/man/Documents/Sycl_Graph_Old/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/p_out_0/"

sys.path.append(Binder_path)
sys.path.append('/usr/local/lib')
from SIR_SBM import *

def get_avg_init_state(dirpath, N):
    avg_init_state = None
    for i in range(N):
        filename = dirpath + "community_traj_" + str(i) + ".csv"
        with open(filename, 'r') as f:
            first_line = f.readline()
            if avg_init_state is None:
                avg_init_state = np.array(first_line.split(",")).astype(float)
            else:
                avg_init_state += np.array(first_line.split(",")).astype(float)
    return avg_init_state / N


def load_data(dirpath, N_sims):
    c_targets = np.genfromtxt(dirpath + "connection_targets_0.csv", delimiter=",")
    c_sources = np.genfromtxt(dirpath + "connection_sources_0.csv", delimiter=",")
    beta_LS = np.genfromtxt(dirpath + "theta_LS.csv", delimiter=",")
    beta_QR = np.genfromtxt(dirpath + "theta_QR.csv", delimiter=",")
    alpha = np.genfromtxt(dirpath + "alpha.csv", delimiter=",")
    community_traj = np.genfromtxt(dirpath + "community_traj_0.csv", delimiter=",")
    N_communities = int(community_traj.shape[1] / 3)
    p_Is_data = np.genfromtxt(dirpath + "p_Is_0.csv", delimiter=",")
    N_connections = p_Is_data.shape[1]
    init_state = get_avg_init_state(dirpath, N_sims)
    N_pop = np.sum(init_state)
    Nt = community_traj.shape[0]
    return c_targets, c_sources, beta_LS, beta_QR, alpha, N_communities, N_connections, init_state, N_pop, Nt
def delta_I(state_s, state_r, p_I, theta):
    return p_I*state_s[0]*state_r[1]/(state_s[0] + state_r[1] + state_s[2])*theta


def solve_single_shoot(c_targets, c_sources, beta, alpha, N_communities, N_connections, init_state, N_pop, Nt, Wu, u_min, u_max):
    
    def community_delta_I(community_idx, c_states, c_p_Is):
        d_I = 0
        state_s = c_states[3*community_idx:3*community_idx+3]
        for i, ct in enumerate(c_targets):
            if (ct == community_idx):
                beta_c = beta[i]
                state_r = c_states[3*int(c_sources[i]):3*int(c_sources[i])+3]
                d_I += delta_I(state_s, state_r, c_p_Is[i], beta_c)
        return d_I
    def community_delta_Rs(c_states):
        return c_states[1::3]*alpha


    N_connections = len(c_targets)
    c_states = MX.sym("c_states", 3*N_communities)
    c_p_Is = MX.sym("c_p_Is", N_connections)
    c_delta_I = []
    c_delta_R = []
    for i in range(N_communities):
        c_delta_I.append(community_delta_I(i, c_states, c_p_Is))
    c_delta_R = community_delta_Rs(c_states)
    c_delta_I = vertcat(*c_delta_I)

    # delta = vertcat(-c_delta_I, c_delta_I - c_delta_R, c_delta_R)

    #zip the delta_I and delta_R together
    delta = []
    for i in range(N_communities):
        delta.append(-c_delta_I[i])
        delta.append(c_delta_I[i] - c_delta_R[i])
        delta.append(c_delta_R[i])
    delta = vertcat(*delta)

    F = Function("f", [c_states, c_p_Is], [delta])

    div_u = 7
    u = MX.sym("u", (int(Nt/div_u)+1, N_connections))
    u_unif = MX.sym("u_unif", (int(Nt/div_u)+1, 1))
    #repeat
    u_uniform = horzcat(*[u_unif for _ in range(N_connections)])


    #generate random p_Is between u_min, u_max

    p_I_test = np.random.rand(Nt, N_connections)*(u_max - u_min) + u_min
    test_state = [init_state]
    deltas = []
    for i in range(Nt):
        test_state.append(test_state[-1] + F(test_state[-1], p_I_test[i,:]))
        deltas.append(F(test_state[-1], p_I_test[i,:]))

    def construct_objective(sym_u):
        f = 0
        state = [DM(init_state)]

        for i in range(Nt):
            u_i = sym_u[int(floor(i/div_u)),:]
            state.append(state[-1] + F(state[-1], u_i))
            inf_sum = 0
            for j in range(N_communities):
                inf_sum += state[i][1+3*j]
            f += inf_sum/N_pop
            for k in range(N_connections):
                f -= Wu/N_pop*(u_i[k] - u_max)
        return f, state
    
    f, state = construct_objective(u)

    w = reshape(u, (u.shape[0]*N_connections, 1))
    f_w_u = Function("f_w_u", [w], [u])
    f_traj = Function("f_traj", [w], [horzcat(*state)])

    ipopt_options = {"ipopt": {"print_level":0,"file_print_level": 5, "tol":1e-8, "max_iter":1000, "output_file":Data_dir + "ipopt_output.txt"}}

    solver = nlpsol("solver", "ipopt", {"x":w, "f":f}, ipopt_options)

    #solve
    res = solver(x0=np.ones(w.shape)*u_min, lbx=u_min*np.ones(w.shape), ubx=u_max*np.ones(w.shape))
    obj_val = res['f'][0][0]
    w_opt = res["x"].full()
    u_opt = f_w_u(w_opt).full()
    traj = f_traj(w_opt).T.full()


    f_uniform, state_unif = construct_objective(u_uniform)

    f_traj_uniform = Function("f_traj_uniform", [u_unif], [horzcat(*state_unif)])

    solver_uniform = nlpsol("solver_uniform", "ipopt", {"x":u_unif, "f":f_uniform}, ipopt_options)

    res_uniform = solver_uniform(x0=np.ones(u_unif.shape)*u_min, lbx=u_min*np.ones(u_unif.shape), ubx=u_max*np.ones(u_unif.shape))

    u_opt_uniform = res_uniform["x"].full()
    traj_uniform = f_traj_uniform(u_opt_uniform).T.full()

    obj_val_uniform = res_uniform['f'][0][0]
    return u_opt, traj, obj_val, u_opt_uniform, traj_uniform, obj_val_uniform


def solution_plot(traj, u_opt, obj_val, Data_dir):
    fig, ax = plt.subplots(4)
    for i in range(3):
        ax[i].plot(traj[:,i::3])
    ax[3].plot(u_opt)
    fig.savefig(Data_dir + "/optimal_community_traj.png")

    fig2, ax2 = plt.subplots(3)
    total_state = np.array([np.sum(traj[:,i::3], axis=1) for i in range(3)]).T
    for i in range(3):
        ax2[i].plot(total_state[:,i])
    fig2.suptitle("Total State, f = " + str(obj_val))
    fig2.savefig(Data_dir + "/total_state.png")

if __name__ == '__main__':
    dirpath = Data_dir + "Graph_0/"

    N_sims = 100
    tau = 0.8
    Wu = 300
    u_max = 1e-2
    u_min = 1e-3
    alpha, theta_LS, theta_QR = regression_on_datasets(dirpath, N_sims, tau, 0)
    # alpha, theta_LS, theta_QR = regression_on_datasets(dirpath, N_sims, tau)

    c_targets, c_sources, beta_LS, beta_QR, alpha, N_communities, N_connections, init_state, N_pop, Nt = load_data(dirpath, N_sims)


    u_opt_LS, traj_LS, f_LS, u_opt_LS_uniform, traj_LS_uniform, f_LS_uniform = solve_single_shoot(c_targets, c_sources, beta_LS, alpha, N_communities, N_connections, init_state, N_pop, Nt, Wu, u_min, u_max)

    u_opt_QR, traj_QR, f_QR, u_opt_QR_uniform, traj_QR_uniform, f_QR_uniform = solve_single_shoot(c_targets, c_sources, beta_QR, alpha, N_communities, N_connections, init_state, N_pop, Nt, Wu, u_min, u_max)

    if not os.path.exists(Data_dir + "LS/"):
       os.makedirs(Data_dir + "LS/")
    if not os.path.exists(Data_dir + "QR/"):
         os.makedirs(Data_dir + "QR/")
    #write all to file
    np.savetxt(Data_dir + "LS/u_opt.csv", u_opt_LS)
    np.savetxt(Data_dir + "LS/traj.csv", traj_LS)
    np.savetxt(Data_dir + "LS/u_opt_uniform.csv", u_opt_LS_uniform)
    np.savetxt(Data_dir + "LS/traj_uniform.csv", traj_LS_uniform)
    np.savetxt(Data_dir + "QR/u_opt.csv", u_opt_QR)
    np.savetxt(Data_dir + "QR/traj.csv", traj_QR)
    np.savetxt(Data_dir + "QR/u_opt_uniform.csv", u_opt_QR_uniform)
    np.savetxt(Data_dir + "QR/traj_uniform.csv", traj_QR_uniform)
    

    solution_plot(traj_LS, u_opt_LS, f_LS, Data_dir + "LS/")
    solution_plot(traj_QR, u_opt_QR, f_QR, Data_dir + "QR/")