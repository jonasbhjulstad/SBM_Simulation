//
// Created by arch on 9/29/22.
//

#ifndef FROLS_FROLS_GRAPH_HPP
#define FROLS_FROLS_GRAPH_HPP

#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <array>
#include <algorithm>
#include <FROLS_Mutex.hpp>
#include <utility>
// #include <range/v3/all.hpp>
#include <limits>

namespace FROLS::Graph
{
    template <typename D>
    struct Vertex
    {
        uint32_t id = std::numeric_limits<uint32_t>::max();
        D data;
        std::shared_ptr<FROLS::mutex> mx;
    };
    template <typename D>
    struct Edge
    {
        D data;
        uint32_t to = std::numeric_limits<uint32_t>::max();
        uint32_t from = std::numeric_limits<uint32_t>::max();
        std::shared_ptr<FROLS::mutex> mx;
    };

    template <typename V, typename E, uint32_t NV, uint32_t NE>
    struct GraphContainer
    {

        GraphContainer(std::array<std::shared_ptr<FROLS::mutex>, NV + 1> &v_mx, std::array<std::shared_ptr<FROLS::mutex>, NE + 1> &e_mx)
        {
            m_data = &_vertices[0];
            for (int i = 0; i < NV + 1; i++)
            {
                _vertices[i].mx = v_mx[i];
            }

            for (int i = 0; i < NE + 1; i++)
            {
                _edges[i].mx = e_mx[i];
            }
            // std::transform(_vertex_mx.begin(), _vertex_mx.end(), _vertices.begin(), [](auto& mx) { return Vertex<V>{0, V{}, mx}; });
            // std::transform(_edge_mx.begin(), _edge_mx.end(), _edges.begin(), [](auto& mx) { return Edge<E>{E{}, 0, 0, mx}; });
        }

    public:
        auto begin()
        {
            return _vertices.begin();
        }
        auto end()
        {
            return &_vertices[N_vertices];
        }
        auto begin() const
        {
            return _vertices.begin();
        }
        auto end() const
        {
            return &_vertices[N_vertices];
        }

        const V &operator[](uint32_t id) const
        {
            return get_vertex_prop(id);
        }

        const V &get_vertex_prop(uint32_t id) const
        {
            return get_vertex(id)->data;
        }

        const Vertex<V> *get_vertex(uint32_t id) const
        {
            assert(_vertices[0].id == 0);
            // find vertex based on index
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [id](Vertex<V> v)
                                    { return v.id == id; });
            assert(p_V != _vertices.end() && "Vertex not found");
            return p_V;
        }

        void assign_vertex(const V &v_data, uint32_t idx)
        {
            // find index of vertex
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [idx](const Vertex<V> &v)
                                    {
                std::lock_guard lock(*v.mx);
                return v.id == idx; });
            assert(p_V != _vertices.end() && "Vertex not found");
            p_V->data = v_data;
        }

        void add_vertex(uint32_t id, const V &v_data)
        {
            assert(N_vertices < NV && "Max number of vertices exceeded");
            std::lock_guard lock(*_vertices[N_vertices].mx);
            _vertices[N_vertices].id = id;
            _vertices[N_vertices].data = v_data;
            ++N_vertices;
        }

        void add_edge(uint32_t from, uint32_t to, const E e_data = {})
        {
            assert(N_edges < NE && "Max number of edges exceeded");
            std::lock_guard lock(*_edges[N_edges].mx);
            _edges[N_edges].from = from;
            _edges[N_edges].to = to;
            _edges[N_edges].data = e_data;
            N_edges++;
        }

        V node_prop(uint32_t id)
        {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                     { 
                                        std::lock_guard lock(*v.mx);
                                        return v.id == id; });
            assert(node != std::end(_vertices) && "Index out of bounds");
            return node->data;
        }

        void assign(uint32_t idx, const V &v_data)
        {
            auto vertex = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                       { 
                std::lock_guard lock(*v.mx);
                return v.id == idx; });
            assert(vertex != std::end(_vertices) && "Index out of bounds");
            vertex->data = v_data;
        }

        void remove_vertex(uint32_t id)
        {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                     { return v.id == id; });
            assert(N_vertices > 0 && "No vertices to remove");
            assert(node != std::end(_vertices) && "Index not found");
            node->id = std::numeric_limits<uint32_t>::max();
            std::for_each(_edges.begin(), _edges.end(), [&](auto &e)
                          {
                std::lock_guard lock(*e.mx);
                if (e.from == id || e.to == id) {
                    e.from = std::numeric_limits<uint32_t>::max();
                    e.to = std::numeric_limits<uint32_t>::max();
                } });

            // sort vertices by id
            std::sort(_vertices.begin(), _vertices.end(), [](const auto &v1, const auto &v2)
                      { std::lock_guard lock1(*v1.mx); std::lock_guard lock2(*v2.mx); return v1.id < v2.id; });
            N_vertices--;
        }

        void remove_edge(uint32_t to, uint32_t from)
        {
            auto edge = std::find_if(_edges.begin(), _edges.end(), [&](const auto &e)
                                     { return e.from == from && e.to == to; });
            assert(N_edges > 0 && "No edges to remove");
            assert(edge != std::end(_edges) && "Index not found");
            edge->from = std::numeric_limits<uint32_t>::max();
            // sort separate edges by from
            std::sort(_edges.begin(), _edges.end(), [](const auto &e1, const auto &e2)
                      { std::lock_guard lock1(*e1.mx); std::lock_guard lock2(*e2.mx); return e1.from < e2.from; });
            N_edges--;
        }

        bool is_in_edge(const Edge<E> &e, uint32_t idx) const
        {
            return (e.to == idx && e.from != std::numeric_limits<uint32_t>::max()) || (e.from != std::numeric_limits<uint32_t>::max() && e.to == idx);
        }

        bool is_valid(const Edge<E> &e) const
        {
            return e.from != std::numeric_limits<uint32_t>::max() && e.to != std::numeric_limits<uint32_t>::max();
        }

        const Vertex<V> *get_neighbor(const Edge<E> &e, uint32_t idx) const
        {
            const FROLS::lock_guard lock(*e.mx);
            uint32_t n_idx = 0;
            if (is_valid(e))
            {
                return e.to == idx ? get_vertex(e.from) : get_vertex(e.to);
            }
            return nullptr;
        }

    protected:
        Vertex<V> *m_data;
        std::array<Vertex<V>, NV + 1> _vertices;
        std::array<Edge<E>, NE + 1> _edges;
        uint32_t N_vertices = 0;
        uint32_t N_edges = 0;
    };

    template <typename V, typename E, uint32_t NV, uint32_t NE>
    struct Graph : public GraphContainer<V, E, NV, NE>
    {
    public:
        using Vertex_t = Vertex<V>;
        using Edge_t = Edge<E>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        static constexpr uint32_t MAX_VERTICES = NV;
        static constexpr uint32_t MAX_EDGES = NE;
        using Base = GraphContainer<V, E, NV, NE>;
        using Base::_edges;
        using Base::_vertices;
        using Base::Base;

        const std::array<const Vertex<V> *, NV> neighbors(uint32_t idx) const
        {
            std::array<const Vertex<V> *, NV> neighbors = {};
            std::for_each(_edges.begin(), _edges.end(), [&, N = 0](const auto e) mutable
                          {
                if (Base::is_in_edge(e, idx))
                {
                    const auto p_nv = this->get_neighbor(e, idx);
                    if (p_nv != nullptr)
                    {
                        neighbors[N] = p_nv;
                        N++;
                    }} });
            return neighbors;
        }
    };

    template <typename V, typename E, uint32_t NV, uint32_t NE>
    Graph(
        const std::array<V, NV> &vertices,
        const std::array<E, NE> &edges) -> Graph<V, E, NV, NE>;

} // FROLS::Graph

#endif // FROLS_FROLS_GRAPH_HPP
