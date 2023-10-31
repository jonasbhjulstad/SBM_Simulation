#include <SBM_Graph/Complete_Graph.hpp>
#include <SBM_Graph/Utils/Math.hpp>
#include <itertools.hpp>
namespace SBM_Graph
{
    std::size_t complete_graph_size(std::size_t N, bool directed, bool self_loops)
    {
        return (N * (N - 1) / 2 + (self_loops ? N : 0)) * (directed ? 2 : 1);
    }

    std::size_t complete_graph_max_edges(std::size_t N, bool self_loops, bool directed)
    {
        return (N * (N - 1) / 2 + (self_loops ? N : 0)) * (directed ? 2 : 1);
    }

    std::vector<Edge_t> complete_graph(size_t N)
    {
        std::vector<Edge_t> edge_list(N * (N - 1) / 2);
        for (auto &&comb : iter::combinations_with_replacement(make_iota(N), 2))
        {
            edge_list.push_back(Edge_t{comb[0], comb[1]});
        }
        return edge_list;
    }
}
