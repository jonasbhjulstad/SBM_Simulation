#include <boost/python.hpp>
#include "SIR_SBM_gt.hpp"

// void graph_convert_bind(GraphInterface& gi, boost::any G)
// {
//     gt_dispatch<>() ([&](auto& _gi, auto _g)
//     {
//         Sycl_Graph::SBM::graph_convert(_gi, _g);
//     }, all_graph_views(), writable_vertex_scalar_properties()) (gi.get_graph_view(), G);
// }

void generate_graph(GraphInterface& gi, const SBM_Graph_t& G)
{
    typedef graph_tool::detail::get_all_graph_views::apply<
    graph_tool::detail::filt_scalar_type, boost::mpl::bool_<false>,
        boost::mpl::bool_<false>, boost::mpl::bool_<false>,
        boost::mpl::bool_<true>, boost::mpl::bool_<true> >::type graph_views;


    run_action<graph_views>()
        (gi, std::bind(gen_graph(), placeholders::_1, N,
                       PythonFuncWrap(deg_sample),
                       no_parallel, no_self_loops,
                       std::ref(rng), verbose, verify))();
}

BOOST_PYTHON_MODULE(libgraph_tool_SIR_SBM)
{
    using namespace boost::python;
    def("graph_convert", &Sycl_Graph::SBM::graph_convert);
}