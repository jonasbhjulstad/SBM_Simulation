//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP

#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Graph/constraints.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <algorithm>
#include <execution>
#include <iostream>
#include <itertools.hpp>
#include <memory>
#include <random>
#include <utility>

namespace Sycl_Graph::Dynamic::Network_Models {
template <Graph_type Graph, typename RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
void random_connect(Graph &G, dType p_ER, RNG &rng, bool directed = false) {
  Static_RNG::uniform_real_distribution d_ER;
  uint32_t N_edges = 0;
  std::vector<uI_t> from;
  std::vector<uI_t> to;
  uint32_t N_edges_max = G.max_edges();

  from.reserve(N_edges_max);
  to.reserve(N_edges_max);
  // std::vector<typename Graph::Edge_Prop_t> edges(N_edges_max);
  for (int i = 0; i < G.N_vertices(); i++) {
    for (int j = i + 1; j < G.N_vertices(); j++) {
      if (d_ER(rng) < p_ER) {
        from.push_back(i);
        N_edges++;
        if (N_edges > N_edges_max) {
          std::cout << "Warning: max edges reached" << std::endl;
          break;
        }
      }
      if (directed) {
        if (d_ER(rng) < p_ER) {
          to.push_back(j);
          N_edges++;
          if (N_edges > N_edges_max) {
            std::cout << "Warning: max edges reached" << std::endl;
            break;
          }
        }
      }
    }
  }
  G.add_edge(to, from);
}

template <Graph_type Graph, std::floating_point dType = float> struct SBM_pack {
  dType p = 0;
  Graph G0;
  Graph G1;
};

template <typename RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
std::vector<std::pair<uI_t, uI_t>>
random_connect(const std::vector<uI_t>& ids_0, const std::vector<uI_t>& ids_1, RNG &rng, dType p,
               bool directed = false) {
  Static_RNG::bernoulli_distribution<dType> d(p);
  uint32_t N_edges = 0;


  std::vector<std::pair<uI_t, uI_t>> edge_ids;
  uint32_t N_expected_edges = ids_0.size()*ids_1.size() * p;
  edge_ids.reserve(N_expected_edges);
  for (auto &&[id_0, id_1] : iter::product(ids_0, ids_1)) {
    if (d(rng)) {
      edge_ids.push_back(std::make_pair(id_0, id_1));
    }
  }

  return edge_ids;
}


typedef std::vector<std::vector<std::pair<uint32_t, uint32_t>>> G_ID_Pairs_t;

template <typename RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
G_ID_Pairs_t
random_connect(const std::vector<std::vector<uI_t>>& ids, std::vector<RNG> &rng,
               const std::vector<std::vector<dType>> &ps, bool directed = false) {
  std::vector<std::pair<std::vector<uI_t>, std::vector<uI_t>>> v_id_pairs;
  v_id_pairs.reserve(ids.size() * ids.size());
  for(auto&& prod : iter::product(ids, ids)) {
   v_id_pairs.push_back(std::make_pair(std::get<0>(prod), std::get<1>(prod)));
  }
  std::vector<dType> ps_flat;
  for(auto&& p : ps) {
    ps_flat.insert(ps_flat.end(), p.begin(), p.end());
  }
  G_ID_Pairs_t edge_ids(ps_flat.size());

  #pragma omp parallel for
  for (int i = 0; i < v_id_pairs.size(); i++) {
    edge_ids[i] = random_connect(v_id_pairs[i].first, v_id_pairs[i].second, rng[i], ps_flat[i], directed);
  }
  return edge_ids;
}



template <typename RNG, std::floating_point dType = float>
G_ID_Pairs_t
random_connect(const std::vector<uint32_t>& N_pops,
               const std::vector<std::vector<dType>> &ps, std::vector<RNG> &rng,
               bool directed = false) {

  typedef std::vector<uint32_t> ID_vec;
  std::vector<std::vector<uint32_t>> ids(N_pops.size());
  std::transform(N_pops.begin(), N_pops.end(), ids.begin(),
  [id_offset = 0](auto &&N_pop) mutable {
    std::vector<uint32_t> id(N_pop);
    std::iota(id.begin(), id.end(), id_offset);
    id_offset += N_pop;
    return id;
  });

  return random_connect(ids, rng, ps, directed);
}

template <Graph_type Graph, typename RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
void random_connect(Graph &G, const std::vector<uI_t> &from_IDs,
                    const std::vector<uI_t> &to_IDs, dType p_ER,
                    RNG rng = Static_RNG::default_rng(std::random_device()())) {
  Static_RNG::uniform_real_distribution d_ER;
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

template <Graph_type Graph, std::floating_point dType = float,
          typename RNG = Static_RNG::default_rng>
Graph generate_erdos_renyi(sycl::queue &q, typename Graph::uI_t NV, dType p_ER,
                           std::vector<typename Graph::uI_t> ids = {},
                           RNG rng = RNG(), typename Graph::uI_t NE = 0) {
  NE = NE == 0 ? 2 * Sycl_Graph::n_choose_k(NV, 2) : NE;

  // get typename as a string
  ids = ids.size() > 0 ? ids : Sycl_Graph::range<typename Graph::uI_t>(0, NV);
  Graph G(q, NV, NE);
  G.add_vertex(ids);
  random_connect(G, p_ER, rng);
  return G;
}

template <typename RNG = Static_RNG::default_rng,
          std::floating_point dType = float>
auto generate_SBM(const std::vector<uint32_t> N_pops,
                  const std::vector<std::vector<dType>> &ps,
                  bool directed = false, std::vector<uint32_t> seeds = {}) {

  if (seeds.size() == 0) {
    for (int i = 0; i < N_pops.size(); i++) {
      seeds.push_back(std::random_device()());
    }
  }
  std::vector<RNG> rngs;
  // initialize vector of rngs
  for (int i = 0; i < N_pops.size() * N_pops.size(); i++) {
    rngs.push_back(RNG(seeds[i]));
  }

  return random_connect(N_pops, ps, rngs, directed);
}

G_ID_Pairs_t generate_planted_partition(uint32_t N_pop, uint32_t N_clusters, float p_in, float p_out, bool directed) {
  std::vector<uint32_t> N_pops(N_clusters, N_pop);
  std::vector<std::vector<float>> ps(N_clusters, std::vector<float>(N_clusters, p_out));
  for (int i = 0; i < N_clusters; i++) {
    ps[i][i] = p_in;
  }
  uint32_t seed = std::random_device()();
  //create rngs
  std::vector<uint32_t> seeds;
  std::random_device rd;
  for (int i = 0; i < N_clusters*N_clusters; i++) {
    seeds.push_back(rd());
  }
  std::vector<Static_RNG::default_rng> rngs;
  for (int i = 0; i < N_clusters*N_clusters; i++) {
    rngs.push_back(Static_RNG::default_rng(seeds[i]));
  }

  return random_connect(N_pops, ps, rngs, directed);
}


} // namespace Sycl_Graph::Dynamic::Network_Models
#endif
