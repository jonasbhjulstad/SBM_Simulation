// graph.cpp
//

#include "graph.hpp"
#define LZZ_INLINE inline
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t bipartite_max_edges (size_t N0, size_t N1)
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                                                 { return N0 * N1; }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  Edgelist_t complete_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1)
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
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
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  Edgelist_t SBM_Graph::flat_edges () const
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                                { return vector_merge(edges); }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 44 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  Vertexlist_t SBM_Graph::flat_vertices () const
#line 44 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                                     { return vector_merge(vertices); }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 45 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t SBM_Graph::N_edges () const
#line 45 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                         {
    return std::accumulate(
        edges.begin(), edges.end(), 0,
        [](auto sum, auto &elem) { return sum + elem.size(); });
  }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 50 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t SBM_Graph::N_vertices () const
#line 50 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                            {
    return std::accumulate(
        vertices.begin(), vertices.end(), 0,
        [](auto sum, auto &elem) { return sum + elem.size(); });
  }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 55 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t SBM_Graph::N_partitions () const
#line 55 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                              { return vertices.size(); }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 56 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t SBM_Graph::N_connections () const
#line 56 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                               { return edges.size(); }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 58 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t SBM_Graph::largest_partition_size () const
#line 58 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                                        {
    return std::max_element(
               vertices.begin(), vertices.end(),
               [](auto &a, auto &b) { return a.size() < b.size(); })
        ->size();
  }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 65 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t SBM_Graph::largest_connection_size () const
#line 65 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                                         {
    return std::max_element(
               edges.begin(), edges.end(),
               [](auto &a, auto &b) { return a.size() < b.size(); })
        ->size();
  }
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 73 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  std::vector <Vertexlist_t> SBM_vertices (size_t N_pop, size_t N_communities)
#line 73 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
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
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  SBM_Graph generate_planted_SBM (size_t N_pop, size_t N_communities, float p_in, float p_out, uint32_t seed)
#line 86 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
                                                           {
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
}
#undef LZZ_INLINE
