#ifndef SBM_GENERATION_HPP
#define SBM_GENERATION_HPP
#include <Sycl_Graph/SBM_types.hpp>

namespace Sycl_Graph::SBM
{

  Edge_List_t
  random_connect(const Node_List_t &to_nodes, const Node_List_t &from_nodes,
                 float p, bool self_loop = true, uint32_t N_threads = 4,
                 uint32_t seed = 47);

  long long n_choose_k(int n, int k);



  SBM_Graph_t random_connect(const std::vector<Node_List_t> &nodelists,
                             float p_in, float p_out, bool self_loop = true, uint32_t N_threads = 4,
                             uint32_t seed = 47);

  // create pybind11 module
  SBM_Graph_t create_SBM(const std::vector<uint32_t> N_pop,
                         float p_in, float p_out,
                         uint32_t N_threads = 4, uint32_t seed = 47);

  SBM_Graph_t create_planted_SBM(uint32_t N_pop, uint32_t N,
                                 float p_in, float p_out,
                                 uint32_t N_threads = 4, uint32_t seed = 47);

  std::vector<SBM_Graph_t> create_planted_SBMs(uint32_t Ng, uint32_t N_pop,
                                               uint32_t N, float p_in, float p_out, uint32_t N_threads = 4, uint32_t seed = 47);

  std::vector<std::vector<float>> generate_p_Is(uint32_t N_community_connections, float p_I_min,
                                                float p_I_max, uint32_t Nt, uint32_t seed = 42);

  std::vector<std::vector<std::vector<float>>> generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, float p_I_min,
                                                             float p_I_max, uint32_t Nt, uint32_t seed = 42);

  std::vector<std::vector<std::vector<std::vector<float>>>> generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, uint32_t Ng, float p_I_min,
                                                                          float p_I_max, uint32_t Nt, uint32_t seed = 42);
}
#endif