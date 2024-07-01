// graph.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_graph_hpp
#define LZZ_SIR_SBM_LZZ_graph_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
#include <cppitertools/combinations_with_replacement.hpp>
#include <SIR_SBM/combination.hpp>
#include <SIR_SBM/random.hpp>
#include <SIR_SBM/vector.hpp>
#include <execution>
#define LZZ_INLINE inline
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  typedef std::pair <uint32_t, uint32_t> Edge_t;
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  typedef std::vector <Edge_t> Edgelist_t;
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  typedef std::vector <uint32_t> Vertexlist_t;
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  size_t bipartite_max_edges (size_t N0, size_t N1);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  Edgelist_t complete_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 26 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  template <typename RNG>
#line 27 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  Edgelist_t generate_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1, float p, RNG & rng);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 39 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  struct SBM_Graph
  {
#line 40 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    std::vector <Edgelist_t> edges;
#line 41 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    std::vector <Vertexlist_t> vertices;
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    Edgelist_t flat_edges () const;
#line 44 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    Vertexlist_t flat_vertices () const;
#line 45 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    size_t N_edges () const;
#line 50 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    size_t N_vertices () const;
#line 55 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    size_t N_partitions () const;
#line 56 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    size_t N_connections () const;
#line 58 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    size_t largest_partition_size () const;
#line 65 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
    size_t largest_connection_size () const;
  };
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 73 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  std::vector <Vertexlist_t> SBM_vertices (size_t N_pop, size_t N_communities);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  SBM_Graph generate_planted_SBM (size_t N_pop, size_t N_communities, float p_in, float p_out, uint32_t seed);
}
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
namespace SIR_SBM
{
#line 26 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  template <typename RNG>
#line 27 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
  Edgelist_t generate_bipartite (Vertexlist_t const & N0, Vertexlist_t const & N1, float p, RNG & rng)
#line 28 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//graph.hpp"
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
