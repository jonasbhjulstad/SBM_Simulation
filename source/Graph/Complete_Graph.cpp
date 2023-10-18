#include <SBM_Simulation/Graph/Complete_Graph.hpp>
#include <itertools.hpp>
#include <SBM_Simulation/Utils/math.hpp>
std::size_t complete_graph_size(std::size_t N, bool directed, bool self_loops)
{
    return (N * (N - 1) / 2 + (self_loops ? N : 0)) * (directed ? 2 : 1);
}

std::size_t complete_graph_max_edges(std::size_t N, bool self_loops, bool directed)
{
    return (N * (N - 1) / 2 + (self_loops ? N : 0)) * (directed ? 2 : 1);
}

std::vector<std::pair<uint32_t, uint32_t>> complete_graph(size_t N)
{
    std::vector<std::pair<uint32_t, uint32_t>> edge_list(N * (N - 1) / 2);
    for (auto &&comb : iter::combinations_with_replacement(make_iota(N), 2))
    {
        edge_list.push_back(std::make_pair(comb[0], comb[1]));
    }
    return edge_list;
}
