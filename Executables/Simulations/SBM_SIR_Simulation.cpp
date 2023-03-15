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
std::pair<Network_t, std::vector<std::vector<std::pair<uint32_t, uint32_t>>>>
 create_SIR_Bernoulli_SBM(const std::vector<size_t> N_pop, const std::vector<float> p_SBM, const float p_I0, const float p_R0 = 0.0, bool undirected = true)
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

    sycl::queue q(sycl::cpu_selector_v,
                  sycl::property::queue::enable_profiling{});
    auto [G, edge_ids_SBM] = generate_SBM<Graph_t, Static_RNG::default_rng>(q, N_pop, p_SBM_reshaped, undirected);
    return std::make_pair(Network_t(G, p_I0, p_R0, edge_ids_SBM), edge_ids_SBM);
}

int main() {

  double p_I0 = 0.1;
  double p_R0 = 0.1;
  static constexpr size_t N_clusters = 2;
  const std::vector<size_t> N_pop = {100, 100};

  std::vector<float> p_SBM = {0.8, 0.1, 0.1, 0.8};
  // create profiling queue
  auto [sir, edge_ids_SBM] = create_SIR_Bernoulli_SBM(N_pop, p_SBM, p_I0, p_R0);
  // generate sir_param
  size_t Nt = 100;
  std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>> sir_param;
  
  const std::array< p_Is = {0.1f, 0.05f, 0.1f};
  for (int i = 0; i < Nt; i++)
  {
    SIR_Bernoulli_SBM_Temporal_Param<N_clusters> param(p_Is);
  }

  sir.initialize();

  auto traj = sir.simulate(Nt, sir_param);
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
}