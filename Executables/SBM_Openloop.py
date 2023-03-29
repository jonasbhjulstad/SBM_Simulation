from casadi import *
import sys
import numpy as np
Project_root = "/home/man/Documents/Sycl_Graph_Old/"
Binder_path = Project_root + "/build/Binders"
Data_dir = Project_root + "/data/SIR_sim/"

def load_data(dirpath):
    c_targets = np.genfromtxt(dirpath + "connection_targets_0.csv", delimiter=",")
    c_sources = np.genfromtxt(dirpath + "connection_sources_0.csv", delimiter=",")
    beta_LS = np.genfromtxt(dirpath + "theta_LS.csv", delimiter=",")
    beta_QR = np.genfromtxt(dirpath + "theta_QR.csv", delimiter=",")
    alpha = np.genfromtxt(dirpath + "alpha.csv", delimiter=",")
    community_traj = np.genfromtxt(dirpath + "community_traj_0.csv", delimiter=",")
    N_communities = int(community_traj.shape[1] / 3)
    p_Is_data = np.genfromtxt(dirpath + "p_Is_0.csv", delimiter=",")
    N_connections = p_Is_data.shape[1]
    init_state = community_traj[0, :]
    N_pop = np.sum(init_state)
    return c_targets, c_sources, beta_LS, beta_QR, alpha, N_communities, N_connections, init_state, N_pop
def delta_I(state_s, state_r, p_I, theta):
    return p_I*state_s[0]*state_r[1]/(state_s[0] + state_r[1] + state_s[2])*theta

if __name__ == '__main__':
    dirpath = Data_dir + "Graph_0/"

    c_targets, c_sources, beta_LS, beta_QR, alpha, N_communities, N_connections, init_state, N_pop = load_data(dirpath)

    def community_delta_I(community_idx, c_states, c_p_Is):
        d_I = 0
        state_s = c_states[3*community_idx:3*community_idx+3]
        for i, ct in enumerate(c_targets):
            if (ct == community_idx):
                beta = beta_LS[i]
                state_r = c_states[3*int(c_sources[i]):3*int(c_sources[i])+3]
                d_I += delta_I(state_s, state_r, c_p_Is[i], beta)
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





    Nt = 40
    u = MX.sym("u", (Nt, N_connections))
    state = [DM(init_state)]
    f = 0
    Wu = 1000
    u_max = 1e-2
    u_min = 1e-3

    #generate random p_Is between u_min, u_max

    p_I_test = np.random.rand(Nt, N_connections)*(u_max - u_min) + u_min
    test_state = [init_state]
    deltas = []
    for i in range(Nt):
        test_state.append(test_state[-1] + F(test_state[-1], p_I_test[i,:]))
        deltas.append(F(test_state[-1], p_I_test[i,:]))
    


    g = []
    for i in range(Nt):
        state.append(state[-1] + F(state[-1], u[i,:]))
        inf_sum = 0
        for j in range(N_communities):
            inf_sum += state[i][1+3*j]
        f += inf_sum/N_pop
        for k in range(N_connections):
            f -= Wu/N_pop*(u[i,k] - u_max)

    w = reshape(u, (Nt*N_connections, 1))
    f_w_u = Function("f_w_u", [w], [u])
    f_traj = Function("f_traj", [w], [horzcat(*state)])

    ipopt_options = {"ipopt": {"print_level":0,"file_print_level": 5, "tol":1e-8, "max_iter":1000, "output_file":Data_dir + "ipopt_output.txt"}}

    solver = nlpsol("solver", "ipopt", {"x":w, "f":f}, ipopt_options)

    #solve
    res = solver(x0=np.ones(w.shape)*u_max, lbx=u_min*np.ones(w.shape), ubx=u_max*np.ones(w.shape))
    w_opt = res["x"].full()
    u_opt = f_w_u(w_opt).full()

    import matplotlib.pyplot as plt
    traj = f_traj(w_opt).T
    fig, ax = plt.subplots(4)
    for i in range(3):
        ax[i].plot(traj[:,i::3])
    ax[3].plot(u_opt)
    plt.show()

    a = 1