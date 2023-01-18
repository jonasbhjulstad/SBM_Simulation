//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP

#include "_Graph_Generation_Sycl_impl.hpp"
#include <CL/sycl.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/random.hpp>
#include <itertools.hpp>
#include <memory>
#include <random>
#include <tinymt/tinymt.h>

namespace Sycl_Graph::Dynamic::Network_Models {
template <typename Graph, typename RNG, typename dType = float>
void random_connect(Graph &G, dType p_ER, RNG &rng) {
  Sycl_Graph::random::uniform_real_distribution d_ER;
  uint32_t N_edges = 0;
  std::vector<typename Graph::uInt_t> from;
  std::vector<typename Graph::uInt_t> to;
  uint32_t N_edges_max = G.NE;

  from.reserve(N_edges_max);
  to.reserve(N_edges_max);
  std::vector<typename Graph::Edge_Prop_t> edges(N_edges_max);
  for (auto &&v_idx : iter::combinations(Sycl_Graph::range(0, G.NV), 2)) {
    if (d_ER(rng) < p_ER) {
      from.push_back(v_idx[0]);
      to.push_back(v_idx[1]);

      N_edges++;
      if (N_edges == G.NE) {
        std::cout << "Warning: max edges reached" << std::endl;
        return;
      }
    }
  }
  G.add_edge(to, from, edges);
}

template <typename Graph, typename RNG, typename dType = float,
          typename uI_t = uint32_t>
void random_connect(Graph &G, const std::vector<uI_t> &from_IDs,
                     const std::vector<uI_t> &to_IDs, dType p_ER,
                     RNG rng = std::mt19937(std::random_device()())) {
  Sycl_Graph::random::uniform_real_distribution d_ER;
  uint32_t N_edges = 0;
  std::vector<typename Graph::uInt_t> from;
  std::vector<typename Graph::uInt_t> to;
  uint32_t N_edges_max = G.NE - G.N_edges();
  from.reserve(N_edges_max);
  to.reserve(N_edges_max);
  std::vector<typename Graph::Edge_Prop_t> edges(N_edges_max);

  // itertools get all combinations of from_IDs and to_IDs
  for (auto &&v_idx : iter::combinations(from_IDs, to_IDs)) {
    if (d_ER(rng) < p_ER) {
      from.push_back(v_idx[0]);
      to.push_back(v_idx[1]);

      N_edges++;
      if (N_edges == N_edges_max) {
        std::cout << "Warning: max edges reached" << std::endl;
        break;
      }
    }
  }
  G.add_edge(to, from, edges);
}

template <typename Graph, typename dType = float, typename uI_t = uint32_t, typename RNG = std::mt19937>
Graph generate_erdos_renyi(sycl::queue &q, uint32_t NV, dType p_ER,
                           std::vector<uI_t> ids = {},
                           RNG rng = std::mt19937(std::random_device()()),
                           uint32_t NE = 0) {
  NE = NE == 0 ? 2 * Sycl_Graph::n_choose_k(NV, 2) : NE;

  ids = ids.size() > 0 ? ids : Sycl_Graph::range(0, NV);
  Graph G(q, NV, NE);
  G.add_vertex(Sycl_Graph::range(0, NV));
  random_connect(G, p_ER, rng);
  return G;
}

} // namespace Sycl_Graph::Dynamic::Network_Models
#endif // FROLS_GRAPH_GENERATION_HPP
