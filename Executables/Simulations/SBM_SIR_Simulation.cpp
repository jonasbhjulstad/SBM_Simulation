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
auto create_SIR_Bernoulli_SBM(sycl::queue& q, const std::vector<size_t> N_pop, const std::vector<float> p_SBM, const float p_I0, const float p_R0 = 0.0, bool undirected = true)
{
    // create profiling queue
    //reshape vector to vector<vector> of size N_pop
    std::vector<std::vector<float>> p_SBM_reshaped(N_pop.size());
    for (size_t i = 0; i < N_pop.size(); i++)
    {
        p_SBM_reshaped[i] = std::vector<float>(N_pop.size());
        for (size_t j = 0; j < N_pop.size(); j++)
        {
            p_SBM_reshaped[i][j] = p_SBM[i * N_pop.size() + j];
        }
    }

    auto [G, edge_ids_SBM] = generate_SBM<Graph_t, Static_RNG::default_rng>(q, N_pop, p_SBM_reshaped, undirected);
    return std::make_tuple(G, Network_t(G, p_I0, p_R0, edge_ids_SBM), edge_ids_SBM);
}

int main() {

  double p_I0 = 0.1;
  double p_R0 = 0.1;
  static constexpr size_t N_clusters = 2;
  const std::vector<size_t> N_pop = {1000, 1000};
  sycl::queue q(sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{});
  std::vector<float> p_SBM = {0.0, 0.0, 0.0, 0.0};
  // std::vector<float> p_SBM = {0.8, 0.1, 0.1, 0.8};
  // create profiling queue
  auto [G, sir, edge_ids_SBM] = create_SIR_Bernoulli_SBM(q, N_pop, p_SBM, p_I0, p_R0);
  // generate sir_param
  size_t Nt = 100;
  // std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>> sir_param(Nt);
  std::vector<float> p_Is = {0.1,0.01,0.1};
  SIR_Bernoulli_SBM_Temporal_Param<>  sir_param;
  std::vector<SIR_Bernoulli_SBM_Temporal_Param<> > sir_param_vec;
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