#include <graph_tool.hh>
#include <Sycl_Graph/SBM_types.hpp>
#include <algorithm>
using namespace graph_tool;
using namespace boost;


namespace Sycl_Graph::SBM
{
    void graph_convert(boost::adj_list<size_t> & gi, const SBM_Graph_t& G)
    {
        std::for_each(G.node_list.begin(), G.node_list.end(), [&](const auto& node)
        {
            add_vertex(gi);
        });

        std::for_each(G.edge_list.begin(), G.edge_list.end(), [&](const auto& edge)
        {
            add_edge(vertex(edge.from, gi), vertex(edge.to, gi), gi);
        });
    }
}