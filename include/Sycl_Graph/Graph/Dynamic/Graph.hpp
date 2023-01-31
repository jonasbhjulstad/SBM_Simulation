#ifndef SYCL_GRAPH_DYNAMIC_GRAPH_HPP
#define SYCL_GRAPH_DYNAMIC_GRAPH_HPP
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <type_traits>
#include <utility>
// #include <Sycl_Graph/execution.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Dynamic/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Dynamic/Vertex_Buffer.hpp>
#include <boost/tokenizer.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <concepts>
namespace Sycl_Graph::Dynamic
{
    template <typename V, typename E, typename uI_t, boost::directed_tag graph_direction = boost::bidirectional>
    struct Graph
    {

        struct Edge_name_tag
        {
            typedef Edge_name_tag kind;
        };

        Graph(uI_t NV, uI_t NE)
            : vertex_buf(NV, props), edge_buf(NE, props), Base_t(vertex_buf, edge_buf) {}

        Graph(const std::vector<Vertex<V, uI_t>> &vertices,
            const std::vector<Edge<E, uI_t>> &edges = {}): vertex_buf(G, vertices), edge_buf(G, edges), Base_t(vertex_buf, edge_buf) {}

        using namespace boost;

        using Vertex_Buffer_t = Vertex_Buffer<V, uI_t>;
        using Edge_Buffer_t = Edge_Buffer<E, uI_t>;

        
        typedef adjacency_list<vecS,vecS, graph_direction, property<vertex_name_type, uI_t>, property<Edge_name_tag, uI_t> Adjlist_t;
        typedef graph_traits<Adjlist_t>::edge_descriptor Adjlist_Edge_t;
        typedef property_map<Adjlist_t, Adjlist_Edge_t>::type Adjlist_Edge_Map_t;
        Adjlist_t G;
    }

}
#endif // SYCL_GRAPH_DYNAMIC_GRAPH_HPP