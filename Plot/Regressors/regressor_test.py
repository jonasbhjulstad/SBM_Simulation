import numpy as np
import matplotlib.pyplot as plt

data_path = "/home/man/Documents/Old_SBM_Simulation/data/SIR_sim/"

def load_data(idx):
    community_traj = np.genfromtxt(data_path + "community_traj_" + str(idx) + ".csv", delimiter=",")
    N_communities = int(community_traj.shape[1] / 3)
    p_Is_data = np.genfromtxt(data_path + "p_Is_" + str(idx) + ".csv", delimiter=",")
    p_Is = np.concatenate([p_Is_data, p_Is_data[:,:-N_communities]], axis=1)
    theta = np.genfromtxt(data_path + "theta_" + str(idx) + ".csv", delimiter=",")
    c_sources = np.genfromtxt(data_path + "connection_sources_" + str(idx) + ".csv", delimiter=",")
    c_targets = np.genfromtxt(data_path + "connection_targets_" + str(idx) + ".csv", delimiter=",")
    alpha = np.genfromtxt(data_path + "alpha_" + str(idx) + ".csv", delimiter=",")
    return p_Is, community_traj, theta, alpha, c_sources, c_targets

def delta_I(state_s, state_r, p_I, theta):
    return p_I*state_s[0]*state_r[1]/(state_s[0] + state_r[1] + state_s[2])*theta


if __name__ == '__main__':

    p_Is, community_traj, theta, alpha, c_sources, c_targets = load_data(0)
    N_communities = int(community_traj.shape[1] / 3)

    def community_delta_I(community_idx, c_states, c_p_Is):
        d_I = 0
        state_s = c_states[3*community_idx:3*community_idx+3]
        for i, ct in enumerate(c_targets):
            if (ct == community_idx):
                th = theta[i]
                state_r = c_states[3*int(c_sources[i]):3*int(c_sources[i])+3]
                d_I += delta_I(state_s, state_r, c_p_Is[i], th)
        return d_I
    def community_delta_Is(c_states, c_p_Is):
        d_Is = np.zeros(N_communities)
        for i in range(N_communities):
            d_Is[i] = community_delta_I(i, c_states, c_p_Is)
        return d_Is

    def community_delta_Rs(c_states):
        d_Rs = np.zeros(N_communities)
        for i in range(N_communities):
            d_Rs[i] = c_states[3*i + 1] * alpha
        return d_Rs

    def next_state(c_states, delta_Is, delta_Rs):
        next_state = np.zeros(c_states.shape)
        for i in range(N_communities):
            next_state[3*i] = c_states[3*i] - delta_Is[i]
            next_state[3*i+1] = c_states[3*i+1] + delta_Is[i] - delta_Rs[i]
            next_state[3*i+2] = c_states[3*i+2] + delta_Rs[i]
        return next_state

    Nt = community_traj.shape[0] - 1

    state = community_traj[0,:]
    traj = np.zeros(community_traj.shape)
    traj[0,:] = state

    for t in range(Nt):
        c_p_Is = p_Is[t,:]
        d_Is = community_delta_Is(state, c_p_Is)
        d_Rs = community_delta_Rs(state)
        state = next_state(state, d_Is, d_Rs)
        traj[t+1,:] = state

    tot_traj = np.array([np.sum(traj[:,i::3], axis=1) for i in range(3)]).T
    tot_community_traj = np.array([np.sum(community_traj[:,i::3], axis=1) for i in range(3)]).T

    fig, ax = plt.subplots(3,3)
    ax[0][0].plot(community_traj[:,::3])
    ax[0][0].set_title("S")
    ax[1][0].plot(community_traj[:,1::3])
    ax[1][0].set_title("I")
    ax[2][0].plot(community_traj[:,2::3])
    ax[2][0].set_title("R")

    ax[0][1].plot(traj[:,::3])
    ax[0][1].set_title("S")
    ax[1][1].plot(traj[:,1::3])
    ax[1][1].set_title("I")
    ax[2][1].plot(traj[:,2::3])
    ax[2][1].set_title("R")

    ax[0][2].plot(tot_community_traj[:,0])
    ax[0][2].set_title("S")
    ax[1][2].plot(tot_community_traj[:,1])
    ax[1][2].set_title("I")
    ax[2][2].plot(tot_community_traj[:,2])
    ax[2][2].set_title("R")

    ax[0][2].plot(tot_traj[:,0])
    ax[1][2].plot(tot_traj[:,1])
    ax[2][2].plot(tot_traj[:,2])



    plt.show()
