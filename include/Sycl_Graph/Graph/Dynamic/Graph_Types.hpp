#ifndef SYCL_GRAPH_DYNAMIC_GRAPH_TYPES_HPP
#define SYCL_GRAPH_DYNAMIC_GRAPH_TYPES_HPP
#include <concepts>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
namespace Sycl_Graph::Dynamic
{

    template <typename T>
    constexpr bool is_boost_graph = std::is_same<boost::adjacency_list<boost::vecS, boost::vecS, typename T::directed_selector,typename T::vertex_property_type, typename T::edge_property_type, typename T::graph_property_type, typename T::edge_list_selector>, T>::value;
    template <typename T>
    concept boost_graph = is_boost_graph<T>;
}

#endif