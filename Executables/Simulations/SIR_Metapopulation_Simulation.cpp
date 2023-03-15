#define SYCL_GRAPH_DEBUG
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Algorithms/Generation/Graph_Generation.hpp>
#include <Sycl_Graph/Network/SIR_Metapopulation/SIR_Metapopulation.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <Static_RNG/distributions.hpp>
#include <algorithm>
#include <filesystem>


static constexpr size_t NV = 1000;
std::vector<uint32_t> N_pop = std::vector<uint32_t>(NV, 1000);
std::vector<Static_RNG::normal_distribution<float>> I0(N_pop.size());
std::vector<Static_RNG::normal_distribution<float>> R0(N_pop.size());
std::vector<float> alpha(N_pop.size(), 0.01);
std::vector<float> node_beta(N_pop.size(), 0.00000005);
std::vector<float> edge_beta(N_pop.size(), 0.00000005);
int main()
{

  std::transform(N_pop.begin(), N_pop.end(), I0.begin(), [](auto x)
                 { return Static_RNG::normal_distribution<float>(x * 0.1, x * 0.01); });

  std::for_each(R0.begin(), R0.end(), [](auto &x)
                { x = Static_RNG::normal_distribution<float>(0, 0); });
  using namespace Sycl_Graph::Sycl::Network_Models;
  using Sycl_Graph::Dynamic::Network_Models::generate_erdos_renyi;
  using namespace Sycl_Graph::Network_Models;
  float p_ER = 0.1;
  //create profiling queue
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
  int seed = 777;
  Static_RNG::default_rng rng;
  auto G = generate_erdos_renyi<SIR_Metapopulation_Graph>(q, NV, p_ER);
  SIR_Metapopulation_Network<> sir(G, N_pop, I0, R0, alpha, node_beta, edge_beta, seed);
  // generate sir_param
  size_t Nt = 100;
  sir.initialize();
  // auto traj = sir.simulate(Nt);

  SIR_Metapopulation_Temporal_Param tp_i{};
  auto traj = sir.simulate_nodes(Nt);
  // print traj

  // write to file
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
  std::ofstream file(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/traj.csv");

  for (int i = 0; i < Nt+1; i++)
  {
    for (int j = 0; j < NV; j++)
    {
      file << traj[i][j].S << ", " << traj[i][j].I << ", " << traj[i][j].R;
      file << (j == NV - 1 ? "" : ", ");
    }
    file << "\n";
  }

  file.close();
}