#pragma once

#include <SIR_SBM/graph/sbm_param.hpp>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>
namespace SIR_SBM {
typedef std::pair<uint32_t, uint32_t> Edge_t;
typedef std::vector<Edge_t> Edgelist_t;
typedef std::vector<uint32_t> Vertexlist_t;

uint32_t bipartite_max_edges(uint32_t N0, uint32_t N1);

Edgelist_t complete_bipartite(const Vertexlist_t &N0, const Vertexlist_t &N1);

Edgelist_t generate_bipartite(const Vertexlist_t &N0, const Vertexlist_t &N1,
                              float p, std::mt19937 &rng);

struct SBM_Graph {
  std::vector<Edgelist_t> edges;
  std::vector<Vertexlist_t> vertices;

  Edgelist_t flat_edges() const;
  Vertexlist_t flat_vertices() const;
  uint32_t N_edges() const;
  uint32_t N_vertices() const;
  std::vector<uint32_t> N_partition_vertices() const;
  uint32_t N_partitions() const;
  uint32_t N_connections() const;

  uint32_t largest_partition_size() const;

  uint32_t largest_connection_size() const;
};

std::vector<Vertexlist_t> SBM_vertices(uint32_t N_pop, uint32_t N_communities);

SBM_Graph generate_planted_SBM(const SBM_Param &p);

} // namespace SIR_SBM