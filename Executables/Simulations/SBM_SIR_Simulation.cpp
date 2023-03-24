#include "Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli_SBM_Types.hpp"
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Algorithms/Generation/Graph_Generation.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli_SBM.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>

using Sycl_Graph::Dynamic::Network_Models::generate_SBM;
using namespace Sycl_Graph::Sycl::Network_Models;
typedef SIR_Bernoulli_SBM_Network Network_t;
typedef typename Network_t::Graph_t Graph_t;

//create pybind11 module
auto create_SIR_Bernoulli_SBM(sycl::queue& q, const std::vector<uint32_t> N_pop, const std::vector<float> p_SBM, const float p_I0, const float p_R0 = 0.0, bool undirected = true)
{
    // create profiling queue
    //reshape vector to vector<vector> of size N_pop
    std::vector<std::vector<float>> p_SBM_reshaped(N_pop.size());
    for (uint32_t i = 0; i < N_pop.size(); i++)
    {
        p_SBM_reshaped[i] = std::vector<float>(N_pop.size());
        for (uint32_t j = 0; j < N_pop.size(); j++)
        {
            p_SBM_reshaped[i][j] = p_SBM[i * N_pop.size() + j];
        }
    }

    auto edge_ids_SBM = generate_SBM<Static_RNG::default_rng>(N_pop, p_SBM_reshaped, undirected);
    


    std::vector<uint32_t> vertex_ids(std::accumulate(N_pop.begin(), N_pop.end(), 0));
    std::iota(vertex_ids.begin(), vertex_ids.end(), 0);
    //flatten edge_ids_SBM to vector of pairs
    std::vector<std::pair<uint32_t, uint32_t>> edge_ids;
    for (uint32_t i = 0; i < edge_ids_SBM.size(); i++)
    {
        for (uint32_t j = 0; j < edge_ids_SBM[i].size(); j++)
        {
            edge_ids.push_back(edge_ids_SBM[i][j]);
        }
    }


    Graph_t G(q, 0, 0);
    G.add_vertex(vertex_ids);
    G.add_edge(edge_ids);

    //create network
    Network_t sir(G, p_I0, p_R0, edge_ids_SBM);

    
    
    
    
    return std::make_pair(G, sir);
}

auto simulate_parallel(std::vector<Network_t>& networks, const std::vector<std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>>>& tp_list)
{
  typedef typename Network_t::Trajectory_t Traj_t;
  std::for_each(networks.begin(), networks.end(), [](auto& network){
    network.initialize();
  });
  std::vector<Traj_t> trajectories(networks.size());
  std::transform(std::execution::par_unseq, networks.begin(), networks.end(), tp_list.begin(), trajectories.begin(), [](auto& network, auto& tp_list){
    return network.simulate(tp_list);
  });
  return trajectories;
}

auto simulate_N_parallel(std::vector<Network_t>& networks, const std::vector<std::vector<std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>>>>& tp_list)
{
  typedef typename Network_t::Trajectory_t Traj_t;
  uint32_t N = tp_list.size();
  std::vector<std::vector<Traj_t>> trajectories(N);
  //reserve
  for (uint32_t i = 0; i < N; i++)
  {
    trajectories[i].reserve(networks.size());
  }
  for (uint32_t i = 0; i < N; i++)
  {
    std::cout << i << " of " << N << std::endl;
    auto traj_i = simulate_parallel(networks, tp_list[i]);
    for(uint32_t j = 0; j < traj_i.size(); j++)
    {
      trajectories[j].push_back(traj_i[j]);
    }
  }
  return trajectories;
}

auto simulate_N_parallel_copied(std::vector<Network_t>& networks, const std::vector<std::vector<std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>>>>& tp_list)
{
  typedef typename Network_t::Trajectory_t Traj_t;

  std::vector<std::vector<Network_t>> networks_list(tp_list.size());
  for (uint32_t i = 0; i < tp_list.size(); i++)
  {
    networks_list[i] = networks;
  }
  
  //print total byte size
  uint32_t total_size = 0;
  for (uint32_t i = 0; i < networks_list.size(); i++)
  {
    for (uint32_t j = 0; j < networks_list[i].size(); j++)
    {
      total_size += networks_list[i][j].byte_size();
    }
  }
  std::cout << "Total byte size: " << total_size << std::endl;

  uint32_t N = tp_list.size();
  auto N_networks = networks.size();
  std::vector<std::vector<Traj_t>> trajectories(N_networks);
    //reserve
    for (uint32_t i = 0; i < N_networks; i++)
    {
      trajectories[i].reserve(N);
    }

    #pragma omp parallel for
    for (uint32_t i = 0; i < N; i++)
    {
      std::cout << i << " of " << N << std::endl;
      auto traj_i = simulate_parallel(networks_list[i], tp_list[i]);
      for(uint32_t j = 0; j < traj_i.size(); j++)
      {
        trajectories[j].push_back(traj_i[j]);
      }
    }
  

  return trajectories;
}

int main() {

  double p_I0 = 0.1;
  double p_R0 = 0.1;
  static constexpr uint32_t N_clusters = 2;
  const std::vector<uint32_t> N_pop = {1000, 1000};
  sycl::queue q(sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{});
  std::vector<float> p_SBM = {0.8, 0.1, 0.1, 8.0};
  // std::vector<float> p_SBM = {0.8, 0.1, 0.1, 0.8};
  // create profiling queue
  auto [G, sir] = create_SIR_Bernoulli_SBM(q, N_pop, p_SBM, p_I0, p_R0);
  auto edge_ids_SBM = sir.SBM_ids;
  // generate sir_param
  uint32_t Nt = 100;
  // std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>> sir_param(Nt);
  std::vector<float> p_Is;
  SIR_Bernoulli_SBM_Temporal_Param<>  sir_param;
  sir_param.p_Is = p_Is;
  typedef std::vector<SIR_Bernoulli_SBM_Temporal_Param<> > Param_t; 
  Param_t sir_param_vec;
  for (int i = 0; i < Nt; i++)
  {
      sir_param_vec.push_back(sir_param);
  }
  sir.initialize();

  auto [traj, group_traj] = sir.simulate_groups(sir_param_vec);
  // print traj
  for (auto &x : traj) {
    std::cout << x[0] << ", " << x[1] << ", " << x[2] << std::endl;
  }

std::vector<Network_t> networks;
for (int i = 0; i < 10; i++)
{
    auto [Gi, sir_i] = create_SIR_Bernoulli_SBM(q, N_pop, p_SBM, p_I0, p_R0);
    sir_i.initialize();
    networks.push_back(sir_i);
}
    uint32_t N_sims = 2;
    std::vector<std::vector<Param_t>> tp_list(N_sims);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < N_sims; j++)
        {
            tp_list[j].push_back(sir_param_vec);
        }
    }

  auto par_res = simulate_N_parallel_copied(networks, tp_list);

  // write to file
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
  std::ofstream file(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) +
                     "/SIR_sim/SBM_traj.csv");

  for (auto &x : traj) {
    file << x[0] << ", " << x[1] << ", " << x[2] << "\n";
  }
  file.close();
  //write to file
  std::ofstream file2(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) +
                     "/SIR_sim/SBM_group_traj.csv");
  for (auto &x : group_traj) {
    for (int i = 0; i < edge_ids_SBM.size(); i++)
    {
        file2 << x[i];
        if (i != edge_ids_SBM.size() - 1)
        {
            file2 << ", ";
        }
    }
    file2 << "\n";
  }
  file2.close();
  return 0;
}