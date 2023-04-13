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
#include <iostream>

using namespace Sycl_Graph::SBM;

int main()
{
  uint32_t N_clusters = 10;
  uint32_t N_pop = 100;
  float p_in = 1.0f;
  float p_out = 0.2f;
  uint32_t N_sims = 100;
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

  auto G =
      create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed);
  uint32_t N_community_connections = G.N_connections;

//   std::vector<uint32_t> cmap(G.node_list.size(), 0);
// //   cmap[0] = 3; cmap[1] = 2; cmap.back() = 1;
//     std::fill(cmap.begin(), cmap.begin() + 10, 3);
//     std::fill(cmap.begin()+10, cmap.begin() + 20, 2);

//     G.remap(cmap);

    return 0;
}