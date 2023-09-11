from SBM_Optimization import *
if __name__ == '__main__':
    p_out = 0.00
    Graph_dir = Data_dir + "/p_out_0.00" + "/Graph_0/"

    N_sims = 2
    tau = 0.9
    Wu = 5000
    u_max = 1e-2
    u_min = 1e-3
    theta_LS, theta_QR = regression_on_datasets(Graph_dir, N_sims, tau, 0)

    connection_community_map_path = Graph_dir + "ccm.csv"

    ccmap = np.genfromtxt(connection_community_map_path, delimiter=",")
    alpha, N_communities, N_connections, init_state, N_pop, Nt = load_data(Graph_dir, N_sims)

    u_opt_LS, traj_LS, f_LS, u_opt_LS_uniform, traj_LS_uniform, f_LS_uniform = solve_single_shoot(ccmap, beta_LS, alpha, N_communities, N_connections, init_state, N_pop, Nt, Wu, u_min, u_max)

    u_opt_QR, traj_QR, f_QR, u_opt_QR_uniform, traj_QR_uniform, f_QR_uniform = solve_single_shoot(ccmap, beta_QR, alpha, N_communities, N_connections, init_state, N_pop, Nt, Wu, u_min, u_max)


    if not os.path.exists(Graph_dir + "LS/"):
       os.makedirs(Graph_dir + "LS/")
    if not os.path.exists(Graph_dir + "QR/"):
         os.makedirs(Graph_dir + "QR/")
    #write all to file
    np.savetxt(Graph_dir + "LS/u_opt.csv", u_opt_LS)
    np.savetxt(Graph_dir + "LS/traj.csv", traj_LS)
    np.savetxt(Graph_dir + "LS/u_opt_uniform.csv", u_opt_LS_uniform)
    np.savetxt(Graph_dir + "LS/traj_uniform.csv", traj_LS_uniform)
    np.savetxt(Graph_dir + "QR/u_opt.csv", u_opt_QR)
    np.savetxt(Graph_dir + "QR/traj.csv", traj_QR)
    np.savetxt(Graph_dir + "QR/u_opt_uniform.csv", u_opt_QR_uniform)
    np.savetxt(Graph_dir + "QR/traj_uniform.csv", traj_QR_uniform)


    solution_plot(traj_LS, u_opt_LS, f_LS, Graph_dir + "LS/")
    solution_plot(traj_QR, u_opt_QR, f_QR, Graph_dir + "QR/")
    u_opt_comparison_plot(traj_LS, u_opt_LS, f_LS, traj_LS_uniform, u_opt_LS_uniform, f_LS_uniform, Graph_dir + "LS/")
    u_opt_comparison_plot(traj_QR, u_opt_QR, f_QR, traj_QR_uniform, u_opt_QR_uniform, f_QR_uniform, Graph_dir + "QR/")
