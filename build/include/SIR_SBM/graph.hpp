// graph.hpp
//

#ifndef LZZ_graph_hpp
#define LZZ_graph_hpp
#include <SIR_SBM/combination.hpp>
#include <SIR_SBM/random.hpp>
#include <SIR_SBM/vector.hpp>
#include <execution>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  typedef std::pair <uint32_t, uint32_t> Edge_t;
}
namespace SIR_SBM
{
  typedef std::vector <Edge_t> Edgelist_t;
}
namespace SIR_SBM
{
  typedef std::vector <uint32_t> Vertexlist_t;
}
namespace SIR_SBM
{
  size_t bipartite_max_edges (size_t N0, size_t N1);
}
namespace SIR_SBM
{
  Edgelist_t complete_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1);
}
namespace SIR_SBM
{
  template <typename RNG>
  Edgelist_t generate_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1, float p, RNG & rng);
}
namespace SIR_SBM
{
  struct SBM_Graph
  {
    std::vector <Edgelist_t> edges;
    std::vector <Vertexlist_t> vertices;
    Edgelist_t flat_edges () const;
    Vertexlist_t flat_vertices () const;
    size_t N_edges () const;
    size_t N_vertices () const;
    size_t N_partitions () const;
    size_t N_connections () const;
    size_t largest_partition_size () const;
    size_t largest_connection_size () const;
  };
}
namespace SIR_SBM
{
  std::vector <Vertexlist_t> SBM_vertices (size_t N_pop, size_t N_communities);
}
namespace SIR_SBM
{
  SBM_Graph generate_planted_SBM (size_t N_pop, size_t N_communities, float p_in, float p_out, uint32_t seed);
}
namespace SIR_SBM
{
  template <typename RNG>
  Edgelist_t generate_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1, float p, RNG & rng)
                                                 {
  if (p == 0.0)
    return {};
  oneapi::dpl::bernoulli_distribution dist(p);
  auto edges = complete_bipartite(N0, N1);
  if ((p < 1.0))
    std::remove_if(edges.begin(), edges.end(),
                   [&dist, &rng](auto elem) { return !dist(rng); });
  return edges;
}
}
#undef LZZ_INLINE
#endif
