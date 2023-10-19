#ifndef SBM_SIMULATION_GRAPH_COMPLETE_GRAPH_HPP
#define SBM_SIMULATION_GRAPH_COMPLETE_GRAPH_HPP
#include <cstdint>
#include <vector>
#include <SBM_Simulation/Graph/Graph_Types.hpp>
std::size_t complete_graph_max_edges(std::size_t N, bool self_loops=true, bool directed = false);
std::size_t complete_graph_size(std::size_t N, bool directed = true, bool self_loops = false);
std::vector<Edge_t> complete_graph(std::size_t N);


#endif
