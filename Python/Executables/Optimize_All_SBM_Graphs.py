from SBM_Routines.SBM_Optimization import *
import json

def read_graph_data(p_dir):
   graph_dirs = get_graph_dirs(p_dir)
   edgelists = []
   ecms = []
   vcms = []
   for g_dir in graph_dirs:
      edgelists.append(np.genfromtxt(g_dir + "/edgelist.csv", delimiter=","))
      ecms.append(np.genfromtxt(g_dir + "/ecm.csv", delimiter=","))
      vcms.append(np.genfromtxt(g_dir + "/vcm.csv", delimiter=","))
   return edgelists, ecms, vcms

def simulate_with_solution(q, p_dir, u_opt, N_sims):
   sim_param = json.load(open(p_dir + "Sim_Param.json"))
   edge_lists, ecms, vcms = read_graph_data(p_dir)
   N_communities = sim_param["N_communities"]
   N_connections = complete_graph_max_edges(N_communities)

   sim_param.output_dir = p_dir + "/LS_Controlled/"
   if not os.path.exists(sim_param.output_dir):
      os.makedirs(sim_param.output_dir)
   sim_param.N_sims = N_sims
   sim_param.N_graphs = len(vcms)
   #write json to file
   p_I_run(q, sim_param, edge_lists, ecms, vcms, N_connections, u_opt)

def read_ccm(fname):
   mat = np.genfromtxt(fname, delimiter=",")
   edge_from = mat[:,0::3]
   edge_to = mat[:,1::3]
   edge_weight = mat[:,2::3]
   return np.concatenate([(ef, et) for ef, et in zip(edge_from.T, edge_to.T)], axis=0), edge_weight


def optimize(Sim_dir, output_dir, N_sims, theta, Wu, u_min, u_max, Nt_per_u = 7):
   print(Sim_dir)

   connection_community_map_path = Sim_dir + "ccm.csv"

   ccmap = np.genfromtxt(connection_community_map_path, delimiter=",", dtype=int)
   if(len(ccmap.shape) == 1):
      ccmap = np.array([ccmap])
   alpha, N_communities, N_connections, init_state, N_pop, Nt = load_data(Sim_dir, N_sims)

   Nu = int(ceil(Nt / Nt_per_u))
   u = MX.sym("u", Nu, N_connections)

   w_opt, traj, f  = solve_single_shoot(ccmap, theta, alpha, N_communities, init_state, Nt, Wu, u, u_min, u_max, output_dir + "/IPOPT.log")
   u_opt = np.reshape(w_opt, (Nu, N_connections))

   u_uniform = MX.sym("u_single", Nu, 1)
   u_opt_uniform, traj_uniform, f_uniform = solve_single_shoot(ccmap, theta, alpha, N_communities, init_state, Nt, Wu, u_uniform, u_min, u_max, output_dir + "/Uniform_IPOPT.log")


   assert np.all(get_total_traj(traj)[0,:] == get_total_traj(traj)[0,:])

   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   #write all to file
   np.savetxt(output_dir + "f.csv", np.array(f))
   np.savetxt(output_dir + "u_opt.csv", u_opt)
   np.savetxt(output_dir + "traj.csv", traj)

   np.savetxt(output_dir + "f_uniform.csv", np.array(f_uniform))
   np.savetxt(output_dir + "u_opt_uniform.csv", u_opt_uniform)
   np.savetxt(output_dir + "traj_uniform.csv", traj_uniform)

   solution_plot(traj, u_opt, f, output_dir)
   u_opt_comparison_plot(traj, u_opt, f, traj_uniform, u_opt_uniform, f_uniform, output_dir)


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

def optimize_sim_dir(sim_dir, N_sims, tau):
      theta_LS, theta_QR, MSE, MAE= regression_on_datasets(sim_dir, params["N_sims"], tau, 0)
      np.savetxt(sim_dir + "/theta_LS.csv", theta_LS)
      np.savetxt(sim_dir + "/theta_QR.csv", theta_QR)
      np.savetxt(sim_dir + "/MSE.csv", MSE)
      np.savetxt(sim_dir + "/MAE.csv", MAE)
      optimize(sim_dir, sim_dir + "/LS/", params["N_sims"], theta_LS, Wu, params["p_I_min"], params["p_I_max"])
      optimize(sim_dir, sim_dir + "/QR/", params["N_sims"], theta_QR, Wu, params["p_I_min"], params["p_I_max"])


if __name__ == '__main__':

   Wu = 500
   tau = .9
   p_dirs = get_p_dirs(Data_dir)
   # for pd in p_dirs[:1]:
   pd = p_dirs[1]
   print(pd)
   graph_dirs = get_graph_dirs(pd)
   params = get_parameters(pd)
   for gd in graph_dirs:
      optimize_sim_dir(gd + "/True_Communities/", params["N_sims"], tau)
      optimize_sim_dir(gd + "/Detected_Communities/", params["N_sims"], tau)
