#ifndef SYCL_GRAPH_INVARIANT_GRAPH_TYPES_HPP
#define SYCL_GRAPH_INVARIANT_GRAPH_TYPES_HPP
#include <concepts>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>

namespace Sycl_Graph::Invariant
{

    template <Sycl_Graph::Base::Edge_type E, Sycl_Graph::Base::Vertex_type _To, Sycl_Graph::Base::Vertex_type _From>
    struct Edge: public E
    {
        typedef _To To;
        typedef _From From;
    };

    template <typename T>
    concept Edge_type = Sycl_Graph::Base::Edge_type<T> && Sycl_Graph::Base::Vertex_type<typename T::To> && Sycl_Graph::Base::Vertex_type<typename T::From>;
}
#endif