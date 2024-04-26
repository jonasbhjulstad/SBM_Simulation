// graph.cpp
//

#include "graph.hpp"
#define LZZ_INLINE inline
namespace SIR_SBM
{
  size_t bipartite_max_edges (size_t N0, size_t N1)
                                                 { return N0 * N1; }
}
namespace SIR_SBM
{
  Edgelist_t complete_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1)
                                                                              {
  Edgelist_t edges(bipartite_max_edges(N0.size(), N1.size()));
  for (int n0 = 0; n0 < N0.size(); n0++) {
    for (int n1 = 0; n1 < N1.size(); n1++) {
      edges[n0 * N0.size() + n1] = {N0[n0], N1[n1]};
    }
  }
  return edges;
}
}
namespace SIR_SBM
{
  Edgelist_t SBM_Graph::flat_edges () const
                                { return vector_merge(edges); }
}
namespace SIR_SBM
{
  Vertexlist_t SBM_Graph::flat_vertices () const
                                     { return vector_merge(vertices); }
}
namespace SIR_SBM
{
  size_t SBM_Graph::N_edges () const
                         {
    return std::accumulate(
        edges.begin(), edges.end(), 0,
        [](auto sum, auto &elem) { return sum + elem.size(); });
  }
}
namespace SIR_SBM
{
  size_t SBM_Graph::N_vertices () const
                            {
    return std::accumulate(
        vertices.begin(), vertices.end(), 0,
        [](auto sum, auto &elem) { return sum + elem.size(); });
  }
}
namespace SIR_SBM
{
  size_t SBM_Graph::N_partitions () const
                              { return vertices.size(); }
}
namespace SIR_SBM
{
  size_t SBM_Graph::N_connections () const
                               { return edges.size(); }
}
namespace SIR_SBM
{
  size_t SBM_Graph::largest_partition_size () const
                                        {
    return std::max_element(
               vertices.begin(), vertices.end(),
               [](auto &a, auto &b) { return a.size() < b.size(); })
        ->size();
  }
}
namespace SIR_SBM
{
  size_t SBM_Graph::largest_connection_size () const
                                         {
    return std::max_element(
               edges.begin(), edges.end(),
               [](auto &a, auto &b) { return a.size() < b.size(); })
        ->size();
  }
}
namespace SIR_SBM
{
  std::vector <Vertexlist_t> SBM_vertices (size_t N_pop, size_t N_communities)
                                                                           {
  std::vector<Vertexlist_t> Vertexlists(N_communities);
  Vertexlist_t vertices(N_pop);
  size_t Vertex_offset = 0;
  for (int i = 0; i < N_communities; i++) {
    std::iota(vertices.begin(), vertices.end(), Vertex_offset);
    Vertexlists[i] = vertices;
    Vertex_offset += N_pop;
  }
  return Vertexlists;
}
}
namespace SIR_SBM
{
  SBM_Graph generate_planted_SBM (size_t N_pop, size_t N_communities, float p_in, float p_out, uint32_t seed)
                                                           {
  SBM_Graph graph;
  graph.vertices = SBM_vertices(N_pop, N_communities);
  auto combs = combinations_with_replacement(N_communities, 2);
  graph.edges.resize(combs.size());
  auto rngs = generate_rngs<oneapi::dpl::ranlux48>(seed, combs.size());

  std::transform(combs.begin(), combs.end(), rngs.begin(), graph.edges.begin(),
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
}
#undef LZZ_INLINE
