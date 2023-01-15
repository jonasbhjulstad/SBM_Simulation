#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Graph/Graph_Generation.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Network/SIR_Metapopulation/SIR_Metapopulation.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/random.hpp>
#include <algorithm>
#include <filesystem>
using namespace Sycl_Graph::random;
std::vector<uint32_t> N_pop = std::vector<uint32_t>(3, 1000);
std::vector<normal_distribution<float>> I0(N_pop.size());
std::vector<normal_distribution<float>> R0(N_pop.size());
std::vector<float> alpha(N_pop.size(), 0.1);
std::vector<float> node_beta(N_pop.size(), 0.001);
std::vector<float> edge_beta(N_pop.size(), 0.001);
int main()
{

  std::transform(N_pop.begin(), N_pop.end(), I0.begin(), [](auto x)
                 { return normal_distribution<float>(x * 0.1, x * 0.01); });

  using namespace Sycl_Graph::Sycl::Network_Models;
  using Sycl_Graph::Dynamic::Network_Models::generate_erdos_renyi;
  using namespace Sycl_Graph::Network_Models;
  float p_ER = 1;
  sycl::queue q;
  int seed = 777;
  uint32_t NV = 3;
  Sycl_Graph::random::default_rng rng;
  auto G = generate_erdos_renyi<SIR_Metapopulation_Graph>(q, NV, p_ER);
  SIR_Metapopulation_Network<> sir(G, N_pop, I0, R0, alpha, node_beta, edge_beta, seed);
  // generate sir_param
  size_t Nt = 100;
  sir.initialize();
  auto traj = sir.simulate(Nt);
  // print traj
  for (auto &x : traj)
  {
    std::cout << x.S << ", " << x.I << ", " << x.R << std::endl;
  }

  // write to file
  std::ofstream file;
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
  file.open(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/traj.csv");

  for (auto &x : traj)
  {
    file << x.S << ", " << x.I << ", " << x.R << "\n";
  }
  file.close();
}