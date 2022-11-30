#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <limits>
#include <vector>
namespace Sycl_Graph
{
    template <typename D, std::unsigned_integral ID_t>
    struct Vertex
    {
        static constexpr ID_t invalid_id = std::numeric_limits<ID_t>::max();
        ID_t id = std::numeric_limits<ID_t>::max();
        D data;
    };
    template <typename D, std::unsigned_integral ID_t>
    struct Edge
    {
        D data;
        ID_t to = std::numeric_limits<ID_t>::max();
        ID_t from = std::numeric_limits<ID_t>::max();
    };

    template <typename Derived, typename V, typename E, std::unsigned_integral uI_t>
    struct GraphContainerBase
    {
        using Vertex_t = Vertex<V, uI_t>;
        using Edge_t = Edge<E, uI_t>;
        // auto begin()
        // {
        //     const auto &derived = static_cast<Derived const &>(*this);
        //     return derived.v_begin();
        // }

        // auto vertex_iterator(uI_t idx)
        // {
        //     const auto &derived = static_cast<Derived const &>(*this);
        //     return derived.vertex_iterator(idx);
        // }

        // auto end()
        // {
        //     const auto &derived = static_cast<Derived const &>(*this);
        //     return derived.v_end();
        // }

        // auto e_begin()
        // {
        //     const auto &derived = static_cast<Derived const &>(*this);
        //     return derived.e_begin();
        // }
        // auto edge_iterator(uI_t idx)
        // {
        //     const auto &derived = static_cast<Derived const &>(*this);
        //     return derived.edge_iterator(idx);
        // }
        // auto e_end()
        // {
        //     const auto &derived = static_cast<Derived const &>(*this);
        //     return derived.e_end();
        // }

        inline bool is_edge_valid(const Edge_t &e) const
        {
            return e.from != std::numeric_limits<uI_t>::max() &&
                   e.to != std::numeric_limits<uI_t>::max();
        }

        inline bool is_in_edge(const Edge_t &e, uI_t idx) const
        {
            return is_edge_valid(e) && (e.from == idx || e.to == idx);
        }

        inline const Vertex_t *get_neighbor(const Edge_t &e, uI_t id) const
        {
            return (e.from == id) ? &(*find(e.to)) : &(*find(e.from));
        }

        uI_t get_max_vertices()
        {
            const auto &derived = static_cast<Derived const &>(*this);
            return derived.get_max_vertices();
        }

        uI_t get_max_edges()
        {
            const auto &derived = static_cast<Derived const &>(*this);
            return derived.get_max_edges();
        }

        V vertex_prop(const std::vector<uI_t> &ids)
        {
            const auto &derived = static_cast<Derived const &>(*this);
            return derived.vertex_prop(ids);
            // auto it = std::find_if(derived.vertices.begin(), derived.vertices.end(),
            //                        [id](const auto &v) { return v.id == id; });
            // if (it != derived.vertices.end()) {
            //   return it->data;
            // }
            // return {};
        }

        auto find(const std::vector<uI_t> &ids) const
        {
            const auto &derived = static_cast<Derived const &>(*this);
            return derived.find(ids);
        }

        void add(const std::vector<Vertex_t>& edges)
        {
            auto &derived = static_cast<Derived &>(*this);
            derived.add(edges);
        }


        void add(const std::vector<uI_t> &id, const std::vector<V> &v_data)
        {
            auto &derived = static_cast<Derived &>(*this);
            derived.add(id, v_data);
        }

        void add(const std::vector<uI_t> &from, const std::vector<uI_t> &to, const std::vector<E> &e_data = E{})
        {
            auto &derived = static_cast<Derived &>(*this);
            derived.add(from, to, e_data);
        }

        void assign(const std::vector<uI_t> &id, const std::vector<V> &v_data)
        {
            auto &derived = static_cast<Derived &>(*this);
            derived.assign(id, v_data);
        }

        void remove(const std::vector<uI_t> &id)
        {
            const auto &derived = static_cast<Derived const &>(*this);
            derived.remove(id);
        }

        void sort_vertices()
        {
            const auto &derived = static_cast<Derived const &>(*this);
            derived.sort_vertices();
        }

        void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from)
        {
            const auto &derived = static_cast<Derived const &>(*this);
            derived.remove(to, from);
        }

        uI_t N_vertices = 0;
        uI_t N_edges = 0;
    };

} // namespace Sycl_Graph
#endif // SYCL_GRAPH_GRAPH_TYPES_HPP