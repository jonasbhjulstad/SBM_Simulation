#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Algorithms/Generation/Graph_Generation.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <Static_RNG/distributions.hpp>
#include <algorithm>
#include <filesystem>

static constexpr size_t NV = 100;
int main()
{

  double p_I0 = 0.1;
  double p_R0 = 0.1;


  using Sycl_Graph::Dynamic::Network_Models::generate_erdos_renyi;
  using namespace Sycl_Graph::Sycl::Network_Models;
  float p_ER = 0.5;
  // create profiling queue
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
  int seed = 777;
  Static_RNG::default_rng rng;
  auto G = generate_erdos_renyi<SIR_Graph>(q, NV, p_ER);
  SIR_Bernoulli_Network sir(G, p_I0, p_R0);
  // generate sir_param
  size_t Nt = 100;
  std::vector<SIR_Bernoulli_Temporal_Param<float>> sir_param(Nt);

  sir.initialize();

  auto traj = sir.simulate(Nt);
  // print traj
  for (auto &x : traj)
  {
    std::cout << x[0] << ", " << x[1] << ", " << x[2] << std::endl;
  }

  // write to file
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
  std::ofstream file(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/traj.csv");

  for (auto &x : traj)
  {
    file << x[0]<< ", " << x[1] << ", " << x[2] << "\n";
  }
  file.close();
}