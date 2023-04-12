#include <boost/python.hpp>
#include "SIR_SBM_gt.hpp"
#include <functional>


// void graph_convert_bind(GraphInterface& gi, boost::any G)
// {
//     gt_dispatch<>() ([&](auto& _gi, auto _g)
//     {
//         Sycl_Graph::SBM::graph_convert(_gi, _g);
//     }, all_graph_views(), writable_vertex_scalar_properties()) (gi.get_graph_view(), G);
// }

// using namespace Sycl_Graph::SBM;
// void graph_convert(GraphInterface& gi, const SBM_Graph_t& G)
// {



//     gt_dispatch<>()

//         ([&](auto& g){ 
//             graph_conv_bind bind(G);
            
//             bind.eval(g);},
//          all_graph_views())

//         (gi.get_graph_view());
// }
using namespace Sycl_Graph::SBM;
BOOST_PYTHON_MODULE(libgraph_tool_SIR_SBM)
{
    using namespace boost::python;
    def("graph_convert", &graph_convert);
}