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
    template <typename V, typename E, std::unsigned_integer uI_t> 
    struct Uniform_Graph
    {
        Uniform_Graph(const std::vector<Vertex<V, uI_t>> &vertices,
                const std::vector<Edge<E, uI_t>> &edges)
            : vertex_buf(vertices), edge_buf(edges) {}
        using Vertex_t = Vertex<V, uI_t>;
        using Edge_t = Edge<E, uI_t>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        using uInt_t = uI_t;
        using Graph_t = Uniform_Graph<V, E, uI_t>;

        Vertex_Buffer<V, uI_t> vertex_buf;
        Edge_Buffer<E, uI_t> edge_buf;
    };

    // template <std::unsigned_integer uI_t, typename ... Vs, typename ... Es>
    // struct Labeled_Graph
    // {
    //     //get number of types of Vs
    //     static constexpr auto N_LABELS_V = sizeof...(Vs);
    //     //get number of types of Es
    //     static constexpr auto N_LABELS_E = sizeof...(Es);

    //     //create N_LABELS_V Vertex_Buffers
    //     std::array<Vertex_Buffer<Vs, uI_t>, N_LABELS_V> vertex_bufs;
    //     //create N_LABELS_E Edge_Buffers
    //     std::array<Edge_Buffer<Es, uI_t>, N_LABELS_E> edge_bufs;

    //     Labeled_Graph(const std::vector<Vertex<Vs, uI_t>> &... vertices,
    //             const std::vector<Edge<Es, uI_t>> &... edges)
    //         : vertex_bufs{vertices...}, edge_bufs{edges...} {}

        
    // };
}
#endif // SYCL_GRAPH_META_GRAPH_HPP