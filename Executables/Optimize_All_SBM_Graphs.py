from SBM_Optimization import *
import json
def optimize(Graph_dir, N_sims, tau, Wu, u_min, u_max):
   print(Graph_dir)
   theta_LS, theta_QR = regression_on_datasets(Graph_dir, N_sims, tau, 0)

   connection_community_map_path = Graph_dir + "ccm.csv"

   ccmap = np.genfromtxt(connection_community_map_path, delimiter=",")
   alpha, N_communities, N_connections, init_state, N_pop, Nt = load_data(Graph_dir, N_sims)

   u_opt_LS, traj_LS, f_LS, u_opt_LS_uniform, traj_LS_uniform, f_LS_uniform = solve_single_shoot(ccmap, theta_LS, alpha, N_communities, N_connections, init_state, N_pop, Nt, Wu, u_min, u_max)

   u_opt_QR, traj_QR, f_QR, u_opt_QR_uniform, traj_QR_uniform, f_QR_uniform = solve_single_shoot(ccmap, theta_QR, alpha, N_communities, N_connections, init_state, N_pop, Nt, Wu, u_min, u_max)


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

def get_p_dirs(base_dir):
      p_dirs = []
      for d in os.listdir(base_dir):
         if (os.path.isdir(base_dir + d) and d.startswith("p_out")):
               p_dirs.append(base_dir + d + "/")
      return p_dirs

def get_graph_dirs(base_dir):
      graph_dirs = []
      for d in os.listdir(base_dir):
         if (os.path.isdir(base_dir + d) and d.startswith("Graph")):
               graph_dirs.append(base_dir + d + "/")
      return graph_dirs
def get_parameters(base_dir):
      #read jsob
      with open(base_dir + "/Sim_Param.json") as f:
         params = json.load(f)
      return params


if __name__ == '__main__':

   Wu = 5000
   tau = .8
   p_dirs = get_p_dirs(Data_dir)
   for pd in p_dirs:
      graph_dirs = get_graph_dirs(pd)
      params = get_parameters(pd)
      for gd in graph_dirs:
         optimize(gd, params["N_sims"], tau, Wu, params["p_I_min"], params["p_I_max"])
