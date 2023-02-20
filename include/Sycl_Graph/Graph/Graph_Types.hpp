#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <limits>
#include <vector>
#include <Sycl_Graph/Math/math.hpp>
namespace Sycl_Graph
{
    template <typename D, typename ID_t>
    struct Vertex
    {
        //sycl device copyable   
        static constexpr ID_t invalid_id = std::numeric_limits<ID_t>::max();
        ID_t id = std::numeric_limits<ID_t>::max();
        D data;
    };

    template <typename D, typename ID_t>
    std::vector<Vertex<D, ID_t>> make_vertices(const std::vector<D>& data, const std::vector<ID_t>& ids)
    {
        std::vector<Vertex<D, ID_t>> vertices(data.size());
        vertices.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            vertices[i] = {ids[i], data[i]};
        }
        return vertices;
    }

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

<<<<<<< HEAD
    template <typename V, std::unsigned_integer uI_t, typename Derived>
=======
    template <typename D, typename ID_t>
    std::vector<Edge<D, ID_t>> make_edges(const std::vector<D>& data, const std::vector<ID_t>& to, const std::vector<ID_t>& from)
    {
        std::vector<Edge<D, ID_t>> edges;
        edges.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            edges.emplace_back(data[i], to[i], from[i]);
        }
        return edges;
    }

    template <typename V, typename uI_t, typename Derived>
>>>>>>> 91d3e036c5389d2c0f22ce10c91c3fd58e099bff
    struct Vertex_Buffer_Base
    {
        uI_t size() const
        {
            return static_cast<const Derived*>(this)->size();
        }


        void add(const std::vector<uI_t>& ids)
        {
            std::vector<V> data(ids.size());
            add(ids, data);
        }
        
        template <typename T = V, typename = std::enable_if_t<!std::is_same<T, uI_t>::value>>
        void add(const std::vector<V>& data)
        {
            add(data, Sycl_Graph::range(0, data.size()));
        }

        void add(const std::vector<uI_t>& ids, const std::vector<V>& data)
        {
            static_cast<Derived*>(this)->add(ids, data);
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

    template <typename E, std::unsigned_integer uI_t, typename Derived>
    struct Edge_Buffer_Base
    {
        uI_t size() const
        {
            return static_cast<const Derived*>(this)->size();
        }
        void add(const std::vector<uI_t>& to, const std::vector<uI_t>& from, const std::vector<E>& data)
        {
            static_cast<Derived*>(this)->add(to, from, data);
        }
        void add(const std::vector<uI_t>& to, const std::vector<uI_t>& from)
        {
            std::vector<E> data(to.size());
            static_cast<Derived*>(this)->add(to, from, data);
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