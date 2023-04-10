#define TBB_DEBUG 1
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/SBM_Generation.hpp>
#include <Sycl_Graph/SIR_SBM_Network.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/SBM_write.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>

using namespace Sycl_Graph::SBM;

int main()
{
  uint32_t N_clusters = 10;
  uint32_t N_pop = 100;
  float p_in = 1.0f;
  float p_out = 0.2f;
  uint32_t N_sims = 1;
  uint32_t Ng = 1;
  // sycl::queue q(sycl::gpu_selector_v);
  std::vector<std::vector<sycl::queue>> qs(Ng);
  for (int i = 0; i < Ng; i++)
  {
    for (int j = 0; j < N_sims; j++)
    {
      qs[i].push_back(sycl::queue(sycl::gpu_selector_v));
    }
  }
  uint32_t Nt = 70;
  uint32_t seed = 47;
  uint32_t N_threads = 10;

  auto Gs =
      create_planted_SBMs(Ng, N_pop, N_clusters, p_in, p_out, N_threads, seed);
  uint32_t N_community_connections = Gs[0].N_connections;

  float p_I_min = 1e-2f;
  float p_I_max = 1e-1f;
  float p_R = 1e-3f;

  std::vector<std::string> output_dirs(Ng);
  std::transform(Gs.begin(), Gs.end(), output_dirs.begin(),
                 [n = 0](const auto &G) mutable
                 {
                   return std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Graph_" + std::to_string(n++) + "/";
                 });

  std::vector<uint32_t> cmap(Gs[0].node_list.size(), 0);
  cmap[2] = 1; cmap[4] = 1; cmap[3] = 1; cmap[0] = 1;



  auto p_I_vec = generate_p_Is(N_community_connections, N_sims,Ng,  p_I_min, p_I_max, Nt, seed);
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");

  std::vector<std::vector<SIR_SBM_Param_t>> params(Ng);
  std::transform(p_I_vec.begin(), p_I_vec.end(), params.begin(),
                 [&](const auto &p_I)
                 {
                   std::vector<SIR_SBM_Param_t> param_vec(N_sims);
                   std::transform(p_I.begin(), p_I.end(), param_vec.begin(), [&](const auto &p)
                                  {
                  SIR_SBM_Param_t param;
                  param.p_R = 1e-1f;
                  param.p_I0 = 0.1f;
                  param.p_R0 = 0.0f;
                  param.p_I = p;
                  return param; });
                   return param_vec;
                 });

  parallel_simulate_to_file(Gs, params, qs, output_dirs, N_sims);

  return 0;
}