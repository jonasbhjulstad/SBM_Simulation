#include <graph_tool.hh>
#include <Sycl_Graph/SBM_types.hpp>
#include <boost/python.hpp>
#include <algorithm>
using namespace graph_tool;
using namespace boost;


namespace Sycl_Graph::SBM
{
    // //template function alias
    struct graph_conv_bind
    {
    template <typename Graph, typename SBM_Graph>
    void operator()(Graph& gi, const SBM_Graph& G)
    {
        // auto gi = gi.get_graph();
        std::for_each(G.node_list.begin(), G.node_list.end(), [&](const auto& node)
        {
            add_vertex(gi);
        });

        std::for_each(G.edge_list.begin(), G.edge_list.end(), [&](const auto& edge)
        {
            add_edge(vertex(edge.first, gi), vertex(edge.second, gi), gi);
        });
        };
    };

    // void graph_convert(GraphInterface& Gi, const SBM_Graph_t& G)
    // {
    //     // typedef mpl::vector<SBM_Graph_t> param_t;
    //     run_action(Gi, graph_conv_bind(), G)(G);
    // }

    // void graph_convert(GraphInterface& Gi, const SBM_Graph_t& G)
    // {
    //     // auto gi = Gi.get_graph();
    //     // GraphInterface& Gi = boost::python::extract<GraphInterface&>(po);

    //     auto gi = Gi.get_graph();
    //     std::for_each(G.node_list.begin(), G.node_list.end(), [&](const auto& node)
    //     {
    //         add_vertex(gi);
    //     });

    //     std::for_each(G.edge_list.begin(), G.edge_list.end(), [&](const auto& edge)
    //     {
    //         add_edge(vertex(edge.first, gi), vertex(edge.second, gi), gi);
    //     });
    // }

}
