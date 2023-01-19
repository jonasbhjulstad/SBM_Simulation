#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <limits>
#include <vector>
namespace Sycl_Graph
{
    template <typename D, typename ID_t>
    struct Vertex
    {
        Vertex() = default;
        //sycl device copyable   
        static constexpr ID_t invalid_id = std::numeric_limits<ID_t>::max();
        ID_t id = std::numeric_limits<ID_t>::max();
        D data;
    };

    template <typename D, typename ID_t>
    struct Edge
    {
        Edge(const D& data, ID_t to, ID_t from)
            : data(data), to(to), from(from) {}
        Edge(ID_t to, ID_t from)
            : to(to), from(from) {}
        D data;
        static constexpr ID_t invalid_id = std::numeric_limits<ID_t>::max();
        ID_t to = invalid_id;
        ID_t from = invalid_id;
    };

    template <typename V, typename uI_t, typename Derived>
    struct Vertex_Buffer_Base
    {
        uI_t size() const
        {
            return static_cast<const Derived*>(this)->size();
        }
        void add(const V* data)
        {
            static_cast<Derived*>(this)->add(data);
        }
        std::vector<Vertex<V*, uI_t>> get_vertices()
        {
            return static_cast<Derived*>(this)->get_vertices();
        }
        void remove(uI_t index)
        {
            static_cast<Derived*>(this)->remove(index);
        }
    };

    template <typename E, typename uI_t, typename Derived>
    struct Edge_Buffer_Base
    {
        uI_t size() const
        {
            return static_cast<const Derived*>(this)->size();
        }
        void add(const E* data, uI_t to, uI_t from)
        {
            static_cast<Derived*>(this)->add(data, to, from);
        }
        std::vector<Edge<E*, uI_t>> get_edges()
        {
            return static_cast<Derived*>(this)->get_edges();
        }

        void remove(uI_t index)
        {
            static_cast<Derived*>(this)->remove(index);
        }
    };

} // namespace Sycl_Graph
#endif // SYCL_GRAPH_GRAPH_TYPES_HPP