#ifndef GRAPH_GENERATION_HPP
#define GRAPH_GENERATION_HPP
#include <vector>
#include <cstdint>

std::vector<std::pair<uint32_t, uint32_t>> random_connect(const std::vector<uint32_t> &to_nodes,
                                                          const std::vector<uint32_t> &from_nodes, float p, uint32_t connection_idx, uint32_t seed);
std::vector<std::vector<std::pair<uint32_t, uint32_t>>> random_connect(const std::vector<std::vector<uint32_t>> &nodelists,
                                                                       float p_in, float p_out, uint32_t seed);
std::tuple<std::vector<std::vector<std::pair<uint32_t, uint32_t>>>,std::vector<std::vector<uint32_t>>> generate_planted_SBM_edges(uint32_t N_pop, uint32_t N_clusters, float p_in, float p_out, uint32_t seed);

std::vector<std::pair<uint32_t, uint32_t>> complete_ccm(uint32_t N_communities);


#endif
