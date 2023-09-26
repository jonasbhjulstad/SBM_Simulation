from SBM_Routines.Path_Config import *
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/usr/local/lib')
def complete_graph_max_edges(N):

    n_choose_2 = lambda n: n*(n-1)/2
    return int(n_choose_2(N) + N)
def get_avg_init_state(Graph_dir, N):
    avg_init_state = None
    for i in range(N):
        filename = Graph_dir + "/Trajectories/community_trajectory_" + str(i) + ".csv"
        with open(filename, 'r') as f:
            first_line = f.readline()
            if avg_init_state is None:
                avg_init_state = np.array(first_line.split(",")).astype(float)
            else:
                avg_init_state += np.array(first_line.split(",")).astype(float)
    return avg_init_state / N
def get_total_traj(community_traj):
    N_communities = int(community_traj.shape[1] / 3)
    return np.array([np.sum(community_traj[:, i::3], axis=1) for i in range(3)]).T

def load_data(Graph_dir, N_sims):
        # c_sources = np.genfromtxt(Graph_dir + "connection_sources_0.csv", delimiter=",")
    alpha = 0.1
    community_traj = np.genfromtxt(Graph_dir + "/Trajectories/community_trajectory_0.csv", delimiter=",")
    N_communities = int(community_traj.shape[1] / 3)
    p_Is_data = np.genfromtxt(Graph_dir + "/p_Is/p_I_0.csv", delimiter=",")
    if (len(p_Is_data.shape) == 1):
        p_Is_data.resize((p_Is_data.shape[0], 1))
    N_connections = p_Is_data.shape[1]
    init_state = get_avg_init_state(Graph_dir, N_sims)
    N_pop = np.sum(init_state)
    Nt = community_traj.shape[0] - 1
    return alpha, N_communities, N_connections, init_state, N_pop, Nt
def delta_I(state_s, state_r, p_I, theta):
    return p_I*state_s[0]*state_r[1]/(state_s[0] + state_r[1] + state_s[2])*theta

def construct_community_ODE(ccmap, beta, alpha, N_communities, N_connections):
    def community_delta_I(community_idx, c_states, c_p_Is):
        d_I = 0
        state_s = c_states[3*community_idx:3*community_idx+3]

        #Forward:
        for i, ct in enumerate(ccmap[:,1]):
            if (ct == community_idx):
                r_idx = int(ccmap[i,0])
                beta_from = beta[i]
                state_r = c_states[3*r_idx:3*r_idx+3]
                d_I += delta_I(state_s, state_r, c_p_Is[i], beta_from)

        return d_I
    def community_delta_Rs(c_states):
        return c_states[1::3]*alpha


    N_connections = ccmap.shape[0]
    c_states = MX.sym("c_states", 3*N_communities)
    c_p_Is = MX.sym("c_p_Is", N_connections)
    # c_p_Is_duped = vertcat(*[c_p_Is for _ in range(2)])
    c_delta_I = []
    c_delta_R = []
    for i in range(N_communities):
        c_delta_I.append(community_delta_I(i, c_states, c_p_Is))
    c_delta_R = community_delta_Rs(c_states)
    c_delta_I = vertcat(*c_delta_I)

    #zip the delta_I and delta_R together
    delta = []
    for i in range(N_communities):
        delta.append(-c_delta_I[i])
        delta.append(c_delta_I[i] - c_delta_R[i])
        delta.append(c_delta_R[i])
    delta = vertcat(*delta)

    F = Function("f", [c_states, c_p_Is], [delta])
    return F

def construct_ODE_trajectory(Nt, sym_u, Nt_per_u, F, init_state, N_connections):
    state = [DM(init_state)]
    for i in range(Nt):
        u_i = sym_u[int(floor(i/Nt_per_u)),:]
        if (u_i.shape[1] == 1):
            u_i = horzcat(*[u_i for _ in range(N_connections)])
        state.append(state[-1] + F(state[-1], u_i))
    return state


def construct_objective_from_ODE(F, init_state, Nt, Wu, u_max, sym_u):
    f = 0
    Nt_per_u = int(ceil(Nt/sym_u.shape[0]))
    N_pop = int(np.sum(init_state))
    N_connections = sym_u.shape[1]
    N_communities = int(init_state.shape[0]/3)
    state = construct_ODE_trajectory(Nt, sym_u, Nt_per_u, F, init_state, N_connections)
    for t in range(Nt):
        inf_sum = 0
        for j in range(N_communities):
            inf_sum += state[t+1][1+3*j]
        f += inf_sum/N_pop
        u_i = sym_u[int(floor(t/Nt_per_u)),:]
        if (u_i.shape[1] == 1):
            u_i = horzcat(*[u_i for _ in range(N_connections)])
        for k in range(N_connections):
            # f += Wu/N_pop*((u_max - u_i[k])**2)
            f -= Wu/N_pop*(u_i[k] - u_max)
    return f, state

def solve_single_shoot(ccmap, beta, alpha, N_communities, init_state, Nt, Wu, u, u_min, u_max, log_fname):

    #get directory of log_fname
    log_dir = "/".join(log_fname.split("/")[:-1]) + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    w = []

    N_connections = ccmap.shape[0]
    if u.shape[1] != N_connections:
        w = u
        u = horzcat(*[u for _ in range(N_connections)])
    else:
        w = reshape(u, (u.shape[0]*N_connections, 1))
    F = construct_community_ODE(ccmap, beta, alpha, N_communities, N_connections)
    f, state = construct_objective_from_ODE(F, init_state, Nt, Wu, u_max, u)

    f_traj = Function("f_traj", [w], [horzcat(*state)])

    ipopt_options = {"ipopt": {"print_level":0,"file_print_level": 5, "tol":1e-8, "max_iter":1000, "output_file": log_fname}, "print_time":0}


    solver = nlpsol("solver", "ipopt", {"x":w, "f":f}, ipopt_options)
    #solve
    res = solver(x0=np.ones(w.shape)*u_max, lbx=u_min*np.ones(w.shape), ubx=u_max*np.ones(w.shape))
    obj_val = res['f'][0][0]
    w_opt = res["x"].full()
    # u_opt = f_w_u(w_opt).full()
    traj = f_traj(w_opt).T.full()

    return w_opt, traj, obj_val


def solution_plot(traj, u_opt, obj_val, Data_dir):
    fig, ax = plt.subplots(4)
    for i in range(3):
        ax[i].plot(traj[:,i::3])
    ax[3].plot(u_opt)
    fig.savefig(Data_dir + "/optimal_community_traj.png")
    plt.close(fig)
    fig2, ax2 = plt.subplots(3)
    total_state = get_total_traj(traj)
    for i in range(3):
        ax2[i].plot(total_state[:,i])
    fig2.suptitle("Total State, f = " + str(obj_val))
    fig2.savefig(Data_dir + "/total_state.png")
    plt.close(fig2)

def u_opt_comparison_plot(traj, u_opt, obj_val, traj_unif, u_unif, obj_unif, Data_dir):
    fig, ax = plt.subplots(4)
    for i in range(3):
        ax[i].plot(traj[:,i::3], 'k', label='Trajectory')
        ax[i].plot(traj_unif[:,i::3], 'r', label='Uniform controlled trajectory')
    ax[3].plot(u_opt, 'k', label='control input')
    ax[3].plot(u_unif, 'r', label='Uniform control input')

    fig.savefig(Data_dir + "/comparison_plot.png")
    plt.close(fig)
    fig2, ax2 = plt.subplots(3)
    total_state = get_total_traj(traj)
    total_state_unif = get_total_traj(traj_unif)
    for i in range(3):
        ax2[i].plot(total_state[:,i])
        ax2[i].plot(total_state_unif[:,i], 'r', label='Uniform controlled trajectory')
    fig2.suptitle("Total State, f = " + str(obj_val) + ", f_uniform = " + str(obj_unif))
    fig2.savefig(Data_dir + "/total_state_comparison.png")
    plt.close(fig2)
