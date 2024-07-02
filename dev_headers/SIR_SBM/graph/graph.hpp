#pragma once
#hdr
#include <cppitertools/combinations_with_replacement.hpp>
#include <SIR_SBM/utils/combination.hpp>
#include <SIR_SBM/utils/random.hpp>
#include <SIR_SBM/common.hpp>
#end
#src
#include <execution>
#end
namespace SIR_SBM {
typedef std::pair<uint32_t, uint32_t> Edge_t;
typedef std::vector<Edge_t> Edgelist_t;
typedef std::vector<uint32_t> Vertexlist_t;

uint32_t bipartite_max_edges(uint32_t N0, uint32_t N1) { return N0 * N1; }

Edgelist_t complete_bipartite(const Vertexlist_t &N0, const Vertexlist_t &N1) {
  Edgelist_t edges(bipartite_max_edges(N0.size(), N1.size()));
  for (int n0 = 0; n0 < N0.size(); n0++) {
    for (int n1 = 0; n1 < N1.size(); n1++) {
      edges[n0 * N0.size() + n1] = {N0[n0], N1[n1]};
    }
  }
  return edges;
}

template <typename RNG>
Edgelist_t generate_bipartite(const Vertexlist_t &N0, const Vertexlist_t &N1,
                              float p, RNG &rng) {
  if (p == 0.0)
    return {};
  oneapi::dpl::bernoulli_distribution dist(p);
  auto edges = complete_bipartite(N0, N1);
  if ((p < 1.0))
    std::remove_if(edges.begin(), edges.end(),
                   [&dist, &rng](auto elem) { return !dist(rng); });
  return edges;
}

struct SBM_Graph {
  std::vector<Edgelist_t> edges;
  std::vector<Vertexlist_t> vertices;

  Edgelist_t flat_edges() const { return vector_merge(edges); }
  Vertexlist_t flat_vertices() const { return vector_merge(vertices); }
  uint32_t N_edges() const {
    return std::accumulate(
        edges.begin(), edges.end(), 0,
        [](auto sum, auto &elem) { return sum + elem.size(); });
  }
  uint32_t N_vertices() const {
    return std::accumulate(
        vertices.begin(), vertices.end(), 0,
        [](auto sum, auto &elem) { return sum + elem.size(); });
  }
  uint32_t N_partitions() const { return vertices.size(); }
  uint32_t N_connections() const { return edges.size(); }

  uint32_t largest_partition_size() const {
    return std::max_element(
               vertices.begin(), vertices.end(),
               [](auto &a, auto &b) { return a.size() < b.size(); })
        ->size();
  }

  uint32_t largest_connection_size() const {
    return std::max_element(
               edges.begin(), edges.end(),
               [](auto &a, auto &b) { return a.size() < b.size(); })
        ->size();
  }
};

std::vector<Vertexlist_t> SBM_vertices(uint32_t N_pop, uint32_t N_communities) {
  std::vector<Vertexlist_t> Vertexlists(N_communities);
  Vertexlist_t vertices(N_pop);
  uint32_t Vertex_offset = 0;
  for (int i = 0; i < N_communities; i++) {
    std::iota(vertices.begin(), vertices.end(), Vertex_offset);
    Vertexlists[i] = vertices;
    Vertex_offset += N_pop;
  }
  return Vertexlists;
}

SBM_Graph generate_planted_SBM(uint32_t N_pop, uint32_t N_communities, float p_in,
                               float p_out, uint32_t seed) {
  SBM_Graph graph;
  graph.vertices = SBM_vertices(N_pop, N_communities);
  auto combs = iter::combinations_with_replacement(make_iota(N_communities), 2);
  
  auto rngs = generate_rngs<oneapi::dpl::ranlux48>(seed, n_choose_k(N_communities, 2));

  std::transform(combs.begin(), combs.end(), rngs.begin(), std::back_inserter(graph.edges),
                 [N_pop, p_in, p_out](auto comb, auto &rng) {
                   Vertexlist_t N0(N_pop);
                   Vertexlist_t N1(N_pop);
                   std::iota(N0.begin(), N0.end(), N_pop * comb[0]);
                   std::iota(N1.begin(), N1.end(), N_pop * comb[1]);
                   float p = comb[0] == comb[1] ? p_in : p_out;
                   return generate_bipartite(N0, N1, p, rng);
                 });
  return graph;
}
} // namespace SIR_SBM