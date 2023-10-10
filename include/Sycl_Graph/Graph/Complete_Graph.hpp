#ifndef SYCL_GRAPH_GRAPH_COMPLETE_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_COMPLETE_GRAPH_HPP
#include <cstdint>
std::size_t complete_graph_max_edges(std::size_t N, bool self_loops=true, bool directed = false);
std::size_t complete_graph_size(size_t N, bool directed = true, bool self_loops = false);
std::vector<std::pair<uint32_t, uint32_t>> complete_graph(size_t N);


#endif
