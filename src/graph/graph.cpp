#include <SIR_SBM/graph/graph.hpp>

#include <SIR_SBM/utils/combination.hpp>
#include <SIR_SBM/utils/random.hpp>
#include <SIR_SBM/utils/numeric.hpp>

#include <cppitertools/combinations_with_replacement.hpp>
#include <execution>

namespace SIR_SBM {
uint32_t bipartite_max_edges(uint32_t N0, uint32_t N1) { return N0 * N1; }

template <typename T>
static T vector_merge(const std::vector<T> &v) {
  T result;
  for (auto &edges : v) {
    result.insert(result.end(), edges.begin(), edges.end());
  }
  return result;
}
Edgelist_t complete_bipartite(const Vertexlist_t &N0, const Vertexlist_t &N1) {
  Edgelist_t edges(bipartite_max_edges(N0.size(), N1.size()));
  for (int n0 = 0; n0 < N0.size(); n0++) {
    for (int n1 = 0; n1 < N1.size(); n1++) {
      edges[n0 * N0.size() + n1] = {N0[n0], N1[n1]};
    }
  }
  return edges;
}

Edgelist_t generate_bipartite(const Vertexlist_t &N0, const Vertexlist_t &N1,
                              float p, std::mt19937 &rng) {
  if (p == 0.0)
    return {};
  oneapi::dpl::bernoulli_distribution dist(p);
  auto edges = complete_bipartite(N0, N1);
  if ((p < 1.0))
    std::remove_if(edges.begin(), edges.end(),
                   [&dist, &rng](auto elem) { return !dist(rng); });
  return edges;
}

Edgelist_t SBM_Graph::flat_edges() const { return vector_merge(edges); }
Vertexlist_t SBM_Graph::flat_vertices() const { return vector_merge(vertices); }
uint32_t SBM_Graph::N_edges() const {
  return std::accumulate(
      edges.begin(), edges.end(), 0,
      [](auto sum, auto &elem) { return sum + elem.size(); });
}
uint32_t SBM_Graph::N_vertices() const {
  return std::accumulate(
      vertices.begin(), vertices.end(), 0,
      [](auto sum, auto &elem) { return sum + elem.size(); });
}
uint32_t SBM_Graph::N_partitions() const { return vertices.size(); }
uint32_t SBM_Graph::N_connections() const { return edges.size(); }

uint32_t SBM_Graph::largest_partition_size() const {
  return std::max_element(vertices.begin(), vertices.end(),
                          [](auto &a, auto &b) { return a.size() < b.size(); })
      ->size();
}

uint32_t SBM_Graph::largest_connection_size() const {
  return std::max_element(edges.begin(), edges.end(),
                          [](auto &a, auto &b) { return a.size() < b.size(); })
      ->size();
}

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

SBM_Graph generate_planted_SBM(uint32_t N_pop, uint32_t N_communities,
                               float p_in, float p_out, uint32_t seed) {
  SBM_Graph graph;
  graph.vertices = SBM_vertices(N_pop, N_communities);
  auto combs = iter::combinations_with_replacement(make_iota(N_communities), 2);

  auto rngs =
      generate_rngs(seed, n_choose_k(N_communities, 2));

  std::transform(combs.begin(), combs.end(), rngs.begin(),
                 std::back_inserter(graph.edges),
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