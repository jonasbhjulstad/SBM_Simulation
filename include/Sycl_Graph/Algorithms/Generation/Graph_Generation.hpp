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

template <Graph_type Graph, typename RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
std::vector<std::pair<uI_t, uI_t>>
random_connect(Graph &G0, Graph &G1, RNG &rng, dType p,
               bool directed = false) {
  Static_RNG::bernoulli_distribution<dType> d(p);
  typedef typename Graph::Edge_t Edge_t;
  uint32_t N_edges = 0;
  std::vector<uI_t> G0_ids = G0.vertex_buf.get_valid_ids();
  std::vector<uI_t> G1_ids = G1.vertex_buf.get_valid_ids();


  std::vector<std::pair<uI_t, uI_t>> edge_ids;
  uint32_t N_expected_edges = G0.N_vertices() * G1.N_vertices() * p;
  edge_ids.reserve(N_expected_edges);
  for (auto &&[id_0, id_1] : iter::product(G0_ids, G1_ids)) {
    if (d(rng)) {
      edge_ids.push_back(std::make_pair(id_0, id_1));
    }
  }

  return edge_ids;
}
template <Graph_type Graph, typename RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
std::pair<Graph, std::vector<std::vector<std::pair<uI_t, uI_t>>>>
random_connect(std::vector<Graph> &Gs,
               const std::vector<std::vector<dType>> &ps, std::vector<RNG> &rng,
               bool directed = false) {

  auto Np = ps.size(); 
  std::vector<std::vector<std::pair<Graph*, Graph*>>> G_combs;
  G_combs.resize(Np);
  //resize each vector to G.size
  for (auto &&G : G_combs) {
    G.resize(Np);
  }
  for (auto comb : iter::product(Sycl_Graph::range(0, G_combs.size()), Sycl_Graph::range(0, G_combs.size()))) {
    auto idx_0 = std::get<0>(comb);
    auto idx_1 = std::get<1>(comb);
    G_combs[idx_0][idx_1] = std::make_pair(&Gs[idx_0], &Gs[idx_1]);
  }
  // for (int i = 0; i < G_combs.size(); i++) {
  //   G_combs[i][i] = std::make_pair(&Gs[i], &Gs[i]);
  // }

  //flatten G_combs
  std::vector<std::pair<Graph*, Graph*>> G_combs_flat;
  for (auto &&G : G_combs) {
    G_combs_flat.insert(G_combs_flat.end(), G.begin(), G.end());
  }

  // flatten ps
  std::vector<dType> ps_flat;
  for (auto &&p : ps) {
    ps_flat.insert(ps_flat.end(), p.begin(), p.end());
  }

  size_t N_G_packs = 0;
  std::vector<std::vector<std::pair<uI_t, uI_t>>> edge_ids;
  for (int i = 0; i < G_combs_flat.size(); i++)
  {
    edge_ids.push_back(random_connect(*(G_combs_flat[i].first), *(G_combs_flat[i].second), rng[i], ps_flat[i], directed));
  }
  // uI_t N_G_pack = std::distance(G_pack.begin(), G_pack.end());

  // std::vector<std::vector<std::pair<uI_t, uI_t>>> edge_ids(N_G_pack);
  // std::transform(std::execution::par_unseq, G_pack.begin(), G_pack.end(),
  //                edge_ids.begin(), [&](auto &&pack) {
  //                  return random_connect(std::get<0>(pack).first, std::get<0>(pack).second, std::get<1>(pack),
  //                                        std::get<2>(pack), directed);
  //                });

  std::vector<uI_t> from;
  std::vector<uI_t> to;
  // get total size of edge_ids
  uI_t N_new_edges = std::accumulate(
      edge_ids.begin(), edge_ids.end(), 0,
      [](auto &&acc, auto &&edge_id) { return acc + edge_id.size(); });
  from.reserve(N_new_edges);
  to.reserve(N_new_edges);

  for (auto &&edge_id : edge_ids) {
    for (auto &&edge : edge_id) {
      from.push_back(edge.first);
      to.push_back(edge.second);
    }
  }

  // get total number of vertices
  uI_t N_cluster_vertices =
      std::accumulate(Gs.begin(), Gs.end(), 0, [](auto &&acc, auto &&G) {
        return acc + G.N_vertices();
      });
  // get total number of edges
  uI_t N_cluster_edges = std::accumulate(
      edge_ids.begin(), edge_ids.end(), 0,
      [](auto &&acc, auto &&edge_id) { return acc + edge_id.size(); });
  uI_t N_edges = N_cluster_edges + N_new_edges;

  Graph G_res(Gs[0].q, 0, 0);
  for (int i = 0; i < Gs.size(); i++)
  {
    G_res += Gs[i];
  }

  G_res.add_edge(from, to);
  return std::make_pair(G_res, edge_ids);
}

template <Graph_type Graph, typename RNG, std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
void random_connect(Graph &G, const std::vector<uI_t> &from_IDs,
                    const std::vector<uI_t> &to_IDs, dType p_ER,
                    RNG rng = std::mt19937(std::random_device()())) {
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

template <Graph_type Graph, typename RNG = Static_RNG::default_rng,
          std::floating_point dType = float,
          std::unsigned_integral uI_t = uint32_t>
auto generate_SBM(sycl::queue &q, const std::vector<uI_t> NVs,
                  const std::vector<std::vector<dType>> &ps,
                  bool directed = false, std::vector<uI_t> seeds = {}) {
  std::vector<Graph> Gs;
  // initialize vector of graphs
  uI_t id_offset = 0;
  for (int i = 0; i < NVs.size(); i++) {
    auto NE = 2 * Sycl_Graph::n_choose_k(NVs[i], 2);
    Gs.push_back(Graph(q, 0, 0));
    Gs[i].vertex_buf.add(Sycl_Graph::range(id_offset, id_offset + NVs[i]));
    id_offset += NVs[i];
  }

  if (seeds.size() == 0) {
    for (int i = 0; i < NVs.size(); i++) {
      seeds.push_back(std::random_device()());
    }
  }
  std::vector<RNG> rngs;
  // initialize vector of rngs
  for (int i = 0; i < NVs.size() * NVs.size(); i++) {
    rngs.push_back(RNG(seeds[i]));
  }

  return random_connect(Gs, ps, rngs, directed);
}

} // namespace Sycl_Graph::Dynamic::Network_Models
#endif
