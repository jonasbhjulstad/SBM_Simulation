#ifndef SYCL_GRAPH_INVARIANT_GRAPH_TYPES_HPP
#define SYCL_GRAPH_INVARIANT_GRAPH_TYPES_HPP
#include <concepts>
#include <Sycl_Graph/Graph/Graph_Types.hpp>

namespace Sycl_Graph::Graph::Invariant
{

    template <Sycl_Graph::Edge_type E, Vertex_type _To, Vertex_type _From>
    struct Edge: public E
    {
        typedef _To To;
        typedef _From From;
    };

    template <typename T>
    concept Edge_type = Edge_type<T> && Vertex_type<typename T::To> && Vertex_type<typename T::From>;
}
#endif