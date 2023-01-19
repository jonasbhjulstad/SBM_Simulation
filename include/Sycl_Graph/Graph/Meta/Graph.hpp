#ifndef SYCL_GRAPH_META_GRAPH_HPP
#define SYCL_GRAPH_META_GRAPH_HPP
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
#include <Sycl_Graph/Graph/Meta/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Meta/Vertex_Buffer.hpp>
#include <type_traits>

namespace Sycl_Graph::Meta
{
    template <typename V, typename E, typename uI_t> 
    struct Meta_Graph
    {
        Graph(const std::vector<Vertex<V, uI_t>> &vertices,
                const std::vector<Edge<E, uI_t>> &edges)
            : vertex_buf(vertices), edge_buf(edges) {}
        using Vertex_t = Vertex<V, uI_t>;
        using Edge_t = Edge<E, uI_t>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        using uInt_t = uI_t;
        using Graph_t = Graph<V, E, uI_t>;

        Vertex_Buffer<V, uI_t> vertex_buf;
        Edge_Buffer<E, uI_t> edge_buf;



    };
}