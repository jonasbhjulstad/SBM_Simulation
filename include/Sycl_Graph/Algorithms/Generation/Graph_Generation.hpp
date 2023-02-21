//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP

#include <Sycl_Graph/Graph/constraints.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <itertools.hpp>
#include <memory>
#include <random>
#include <tinymt/tinymt.h>

namespace Sycl_Graph::Dynamic::Network_Models {
template <Graph_type Graph, Static_RNG::rng_type RNG, std::floating_point dType = float>
void random_connect(Graph &G, dType p_ER, RNG &rng) {
  Static_RNG::distributions::uniform_real_distribution d_ER;
  uint32_t N_edges = 0;
  std::vector<typename Graph::uInt_t> from;
  std::vector<typename Graph::uInt_t> to;
  uint32_t N_edges_max = G.N_edges();

  from.reserve(N_edges_max);
  to.reserve(N_edges_max);
  // std::vector<typename Graph::Edge_Prop_t> edges(N_edges_max);
  for (auto &&v_idx : iter::combinations(Sycl_Graph::range<uint32_t>(0, G.N_vertices()), 2)) {
    if (d_ER(rng) < p_ER) {
      from.push_back(v_idx[0]);
      to.push_back(v_idx[1]);

      N_edges++;
      if (N_edges == G.N_edges()) {
        std::cout << "Warning: max edges reached" << std::endl;
        return;
      }
    }
  }
  G.add_edge(to, from);
}

template <Graph_type Graph, Static_RNG::rng_type RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
void random_connect(Graph &G, const std::vector<uI_t> &from_IDs,
                     const std::vector<uI_t> &to_IDs, dType p_ER,
                     RNG rng = std::mt19937(std::random_device()())) {
  Static_RNG::distributions::uniform_real_distribution d_ER;
  uint32_t N_edges = 0;
  std::vector<typename Graph::uInt_t> from;
  std::vector<typename Graph::uInt_t> to;
  uint32_t N_edges_max = G.N_edges() - G.N_edges();
  from.reserve(N_edges_max);
  to.reserve(N_edges_max);
  std::vector<typename Graph::Edge_Prop_t> edges(N_edges_max);

  // itertools get all combinations of from_IDs and to_IDs
  
  for (auto &&[from_id, to_id] : iter::product(from_IDs, to_IDs)) {
    if (d_ER(rng) < p_ER) {
      from.push_back(from_id);
      to.push_back(to_id);

      N_edges++;
      if (N_edges == N_edges_max) {
        std::cout << "Warning: max edges reached" << std::endl;
        break;
      }
    }
  }
  G.add_edge(to, from, edges);
}

template <Graph_type Graph, std::floating_point dType = float, std::unsigned_integral uI_t = uint32_t, Static_RNG::rng_type = std::mt19937>
Graph generate_erdos_renyi(sycl::queue &q, uI_t NV, dType p_ER,
                           std::vector<uI_t> ids = {},
                           RNG rng = std::mt19937(std::random_device()()),
                           uint32_t NE = 0) {
  NE = NE == 0 ? 2 * Sycl_Graph::n_choose_k(NV, 2) : NE;

  ids = ids.size() > 0 ? ids : Sycl_Graph::range<uint32_t>(0, NV);
  Graph G(q, NV+1, NE);
  G.add_vertex(ids);
  random_connect(G, p_ER, rng);
  return G;
}



} // namespace Sycl_Graph::Dynamic::Network_Models
#endif // FROLS_GRAPH_GENERATION_HPP
