//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_FROLS_GRAPH_HPP
#define SYCL_GRAPH_FROLS_GRAPH_HPP

#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <array>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <Data_Containers.hpp>
#include <limits>

namespace Sycl::Graph
{


    template <typename D, std::unsigned_integral ID_t>
    struct Vertex
    {
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


    template <typename V, typename E,  std::unsigned_integral uI_t, uI_t NV, uI_t NE>
    struct ArrayGraph
    {
        static constexpr uI_t NV_MAX = NV;
        static constexpr uI_t NE_MAX = NE;
        const uI_t NV_max;
        const uI_t NE_max;

        using Vertex_t = Vertex<V, uI_t>;
        using Edge_t = Edge<E, uI_t>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        std::array<Vertex_t, NV + 1> _vertices;
        std::array<Edge_t, NE + 1> _edges;
        uI_t N_vertices = 0;
        uI_t N_edges = 0;

        ArrayGraph(std::array<std::mutex *, NV + 1> &v_mx, std::array<std::mutex *, NE + 1> &e_mx) : NV_max(v_mx.size()), NE_max(e_mx.size())
        {

            std::for_each(v_mx.begin(), v_mx.end(), [&, n = 0](auto &mx) mutable
                          {
                _vertices[n].mx = mx;
                n++; });
            std::for_each(e_mx.begin(), e_mx.end(), [&, n = 0](auto &mx) mutable
                          {
                _edges[n].mx = mx;
                n++; });
        }

        ArrayGraph<V, E, NV, NE> operator=(const ArrayGraph<V, E, NV, NE> &other)
        {
            _vertices = other._vertices;
            _edges = other._edges;
            N_vertices = other.N_vertices;
            N_edges = other.N_edges;
            std::for_each(_vertices.begin(), _vertices.end(), [&](auto &v)
                          { v.mx = std::make_shared<std::mutex>(); });
            std::for_each(_edges.begin(), _edges.end(), [&](auto &e)
                          { e.mx = std::make_shared<std::mutex>(); });

            return *this;
        }

        auto begin()
        {
            return std::begin(_vertices);
        }
        auto end()
        {
            return std::begin(_vertices) + N_vertices;
        }

        const Vertex_Prop_t &operator[](uI_t id) const
        {
            return get_vertex_prop(id);
        }

        const Vertex_Prop_t &get_vertex_prop(uI_t id) const
        {
            return get_vertex(id)->data;
        }

        const Vertex_t *get_vertex(uI_t id) const
        {
            assert(_vertices[0].id == 0);
            // find vertex based on index
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [id](Vertex_t v)
                                    { return v.id == id; });
            assert(p_V != _vertices.end() && "Vertex not found");
            return p_V;
        }

        void assign_vertex(const Vertex_Prop_t &v_data, uI_t idx)
        {
            // find index of vertex
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [idx](const Vertex_t &v)
                                    {
                std::lock_guard lock(*v.mx);
                return v.id == idx; });
            assert(p_V != _vertices.end() && "Vertex not found");
            p_V->data = v_data;
        }

        void add_vertex(uI_t id, const Vertex_Prop_t &v_data)
        {
            assert(N_vertices < NV_max && "Max number of vertices exceeded");
            std::lock_guard lock(*_vertices[N_vertices].mx);
            _vertices[N_vertices].id = id;
            _vertices[N_vertices].data = v_data;
            ++N_vertices;
        }

        void add_edge(uI_t from, uI_t to, const Edge_Prop_t e_data = {})
        {
            assert(N_edges < NE_max && "Max number of edges exceeded");
            std::lock_guard lock(*_edges[N_edges].mx);
            _edges[N_edges].from = from;
            _edges[N_edges].to = to;
            _edges[N_edges].data = e_data;
            N_edges++;
        }

        Vertex_Prop_t node_prop(uI_t id)
        {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                     { 
                                        std::lock_guard lock(*v.mx);
                                        return v.id == id; });
            assert(node != std::end(_vertices) && "Index out of bounds");
            return node->data;
        }

        void assign(uI_t idx, const Vertex_Prop_t &v_data)
        {
            auto vertex = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                       { 
                std::lock_guard lock(*v.mx);
                std::cout << "Locking " << v.id << std::endl;

                return v.id == idx; });
            assert(vertex != std::end(_vertices) && "Index out of bounds");
            vertex->data = v_data;
        }

        void remove_vertex(uI_t id)
        {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                     { return v.id == id; });
            assert(N_vertices > 0 && "No vertices to remove");
            assert(node != std::end(_vertices) && "Index not found");
            node->id = std::numeric_limits<uI_t>::max();
            std::for_each(_edges.begin(), _edges.end(), [&](auto &e)
                          {
                std::lock_guard lock(*e.mx);
                if (e.from == id || e.to == id) {
                    e.from = std::numeric_limits<uI_t>::max();
                    e.to = std::numeric_limits<uI_t>::max();
                } });

            // sort vertices by id
            std::sort(_vertices.begin(), _vertices.end(), [](const auto &v1, const auto &v2)
                      { std::lock_guard lock1(*v1.mx); std::lock_guard lock2(*v2.mx); return v1.id < v2.id; });
            N_vertices--;
        }

        void remove_edge(uI_t to, uI_t from)
        {
            auto edge = std::find_if(_edges.begin(), _edges.end(), [&](const auto &e)
                                     { return e.from == from && e.to == to; });
            assert(N_edges > 0 && "No edges to remove");
            assert(edge != std::end(_edges) && "Index not found");
            edge->from = std::numeric_limits<uI_t>::max();
            // sort separate edges by from
            std::sort(_edges.begin(), _edges.end(), [](const auto &e1, const auto &e2)
                      { std::lock_guard lock1(*e1.mx); std::lock_guard lock2(*e2.mx); return e1.from < e2.from; });
            N_edges--;
        }

        bool is_in_edge(const Edge_t &e, uI_t idx) const
        {
            return (e.to == idx && e.from != std::numeric_limits<uI_t>::max()) || (e.from != std::numeric_limits<uI_t>::max() && e.to == idx);
        }

        bool is_valid(const Vertex_t &v) const
        {
            return v.id != std::numeric_limits<uI_t>::max();
        }

        bool is_valid(const Edge_t &e) const
        {
            return e.from != std::numeric_limits<uI_t>::max() && e.to != std::numeric_limits<uI_t>::max();
        }

        const Vertex_t *get_neighbor(const Edge_t &e, uI_t idx) const
        {
            const Sycl::Graph::lock_guard lock(*e.mx);
            uI_t n_idx = 0;
            if (is_valid(e))
            {
                return e.to == idx ? get_vertex(e.from) : get_vertex(e.to);
            }
            return nullptr;
        }

        const std::array<Vertex_t *, NV + 1> neighbors(ID_t idx) const
        {
            std::array<Vertex_t *, NV + 1> neighbors = {};
            if (neighbors.size() < NV_max)
            {
                neighbors.resize(NV_max);
            }
            std::for_each(_edges.begin(), _edges.end(), [&, N = 0](const auto e) mutable
                          {
                if (is_in_edge(e, idx))
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

} // Sycl::Graph::Graph

#endif // FROLS_FROLS_GRAPH_HPP
