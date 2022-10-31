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
#include <type_traits>
// #include <range/v3/all.hpp>
#include <limits>

namespace FROLS::Graph
{
    template <typename D>
    struct Vertex
    {
        uint32_t id = std::numeric_limits<uint32_t>::max();
        D data;
        std::shared_ptr<std::mutex> mx;
    };
    template <typename D>
    struct Edge
    {
        D data;
        uint32_t to = std::numeric_limits<uint32_t>::max();
        uint32_t from = std::numeric_limits<uint32_t>::max();
        std::shared_ptr<std::mutex> mx;
    };

    // template <typename V, typename E>
    // struct VectorGraphContainer
    // {
    //     const uint32_t NV_max;
    //     const uint32_t NE_max;

    //     using Vertex_t = Vertex<V>;
    //     using Edge_t = Edge<E>;
    //     using Vertex_Prop_t = V;
    //     using Edge_Prop_t = E;
    //     std::vector<Vertex_t> _vertices;
    //     std::vector<Edge_t> _edges;
    //     uint32_t N_vertices = 0;
    //     uint32_t N_edges = 0;

    //     VectorGraphContainer(std::vector<std::shared_ptr<std::mutex>> &v_mx, std::vector<std::shared_ptr<std::mutex>> &e_mx) : NV_max(v_mx.size()), NE_max(e_mx.size())
    //     {

    //             _vertices.resize(NV_max);
    //             _edges.resize(NE_max);
    //         std::for_each(v_mx.begin(), v_mx.end(), [&, n = 0](auto &mx) mutable
    //                       {
    //             _vertices[n].mx = mx;
    //             n++; });
    //         std::for_each(e_mx.begin(), e_mx.end(), [&, n = 0](auto &mx) mutable
    //                       {
    //             _edges[n].mx = mx;
    //             n++; });

    //     }

    //     auto begin()
    //     {
    //         return std::begin(_vertices);
    //     }
    //     auto end()
    //     {
    //         return std::begin(_vertices) + N_vertices;
    //     }
    // };

    template <typename V, typename E, uint32_t NV, uint32_t NE>
    struct ArrayGraph
    {
        static constexpr uint32_t NV_MAX = NV;
        static constexpr uint32_t NE_MAX = NE;
        const uint32_t NV_max;
        const uint32_t NE_max;

        using Vertex_t = Vertex<V>;
        using Edge_t = Edge<E>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        std::array<Vertex_t, NV + 1> _vertices;
        std::array<Edge_t, NE + 1> _edges;
        uint32_t N_vertices = 0;
        uint32_t N_edges = 0;

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

        const Vertex_Prop_t &operator[](uint32_t id) const
        {
            return get_vertex_prop(id);
        }

        const Vertex_Prop_t &get_vertex_prop(uint32_t id) const
        {
            return get_vertex(id)->data;
        }

        const Vertex_t *get_vertex(uint32_t id) const
        {
            assert(_vertices[0].id == 0);
            // find vertex based on index
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [id](Vertex_t v)
                                    { return v.id == id; });
            assert(p_V != _vertices.end() && "Vertex not found");
            return p_V;
        }

        void assign_vertex(const Vertex_Prop_t &v_data, uint32_t idx)
        {
            // find index of vertex
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [idx](const Vertex_t &v)
                                    {
                std::lock_guard lock(*v.mx);
                return v.id == idx; });
            assert(p_V != _vertices.end() && "Vertex not found");
            p_V->data = v_data;
        }

        void add_vertex(uint32_t id, const Vertex_Prop_t &v_data)
        {
            assert(N_vertices < NV_max && "Max number of vertices exceeded");
            std::lock_guard lock(*_vertices[N_vertices].mx);
            _vertices[N_vertices].id = id;
            _vertices[N_vertices].data = v_data;
            ++N_vertices;
        }

        void add_edge(uint32_t from, uint32_t to, const Edge_Prop_t e_data = {})
        {
            assert(N_edges < NE_max && "Max number of edges exceeded");
            std::lock_guard lock(*_edges[N_edges].mx);
            _edges[N_edges].from = from;
            _edges[N_edges].to = to;
            _edges[N_edges].data = e_data;
            N_edges++;
        }

        Vertex_Prop_t node_prop(uint32_t id)
        {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                     { 
                                        std::lock_guard lock(*v.mx);
                                        return v.id == id; });
            assert(node != std::end(_vertices) && "Index out of bounds");
            return node->data;
        }

        void assign(uint32_t idx, const Vertex_Prop_t &v_data)
        {
            auto vertex = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                       { 
                std::lock_guard lock(*v.mx);
                std::cout << "Locking " << v.id << std::endl;

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

        bool is_in_edge(const Edge_t &e, uint32_t idx) const
        {
            return (e.to == idx && e.from != std::numeric_limits<uint32_t>::max()) || (e.from != std::numeric_limits<uint32_t>::max() && e.to == idx);
        }

        bool is_valid(const Vertex_t &v) const
        {
            return v.id != std::numeric_limits<uint32_t>::max();
        }

        bool is_valid(const Edge_t &e) const
        {
            return e.from != std::numeric_limits<uint32_t>::max() && e.to != std::numeric_limits<uint32_t>::max();
        }

        const Vertex_t *get_neighbor(const Edge_t &e, uint32_t idx) const
        {
            const FROLS::lock_guard lock(*e.mx);
            uint32_t n_idx = 0;
            if (is_valid(e))
            {
                return e.to == idx ? get_vertex(e.from) : get_vertex(e.to);
            }
            return nullptr;
        }

        const std::array<Vertex_t *, NV + 1> neighbors(uint32_t idx) const
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

    template <typename V, typename E>
    struct VectorGraph
    {
        const uint32_t NV_max;
        const uint32_t NE_max;

        using Vertex_t = Vertex<V>;
        using Edge_t = Edge<E>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        std::vector<Vertex_t> _vertices;
        std::vector<Edge_t> _edges;
        uint32_t N_vertices = 0;
        uint32_t N_edges = 0;

        VectorGraph(std::vector<std::shared_ptr<std::mutex>> &v_mx, std::vector<std::shared_ptr<std::mutex>> &e_mx) : NV_max(v_mx.size()), NE_max(e_mx.size())
        {
            _vertices.resize(NV_max);
            _edges.resize(NE_max);
            std::for_each(v_mx.begin(), v_mx.end(), [&, n = 0](auto &mx) mutable
                          {
                _vertices[n].mx = mx;
                n++; });
            std::for_each(e_mx.begin(), e_mx.end(), [&, n = 0](auto &mx) mutable
                          {
                _edges[n].mx = mx;
                n++; });
        }


        VectorGraph(const std::vector<Vertex_t>& vertices, const std::vector<Edge_t>& edges) : _vertices(vertices), _edges(edges), NV_max(vertices.size()), NE_max(edges.size())
        {
        }


        VectorGraph operator=(const VectorGraph &other)
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
        const Vertex_Prop_t &operator[](uint32_t id) const
        {
            return get_vertex_prop(id);
        }

        const Vertex_Prop_t &get_vertex_prop(uint32_t id) const
        {
            return get_vertex(id)->data;
        }

        const Vertex_t *get_vertex(uint32_t id) const
        {
            assert(_vertices[0].id == 0);
            // find vertex based on index
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [id](Vertex_t v)
                                    { return v.id == id; });
            assert(p_V != _vertices.end() && "Vertex not found");
            return (Vertex_t *)&(*p_V);
        }

        void assign_vertex(const Vertex_Prop_t &v_data, uint32_t idx)
        {
            // find index of vertex
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [idx](const Vertex_t &v)
                                    {
                std::lock_guard lock(*v.mx);
                return v.id == idx; });
            assert(p_V != _vertices.end() && "Vertex not found");
            p_V->data = v_data;
        }

        void add_vertex(uint32_t id, const Vertex_Prop_t &v_data)
        {
            assert(N_vertices < NV_max && "Max number of vertices exceeded");
            std::lock_guard lock(*_vertices[N_vertices].mx);
            _vertices[N_vertices].id = id;
            _vertices[N_vertices].data = v_data;
            ++N_vertices;
        }

        void add_edge(uint32_t from, uint32_t to, const Edge_Prop_t e_data = {})
        {
            assert(N_edges < NE_max && "Max number of edges exceeded");
            std::lock_guard lock(*_edges[N_edges].mx);
            _edges[N_edges].from = from;
            _edges[N_edges].to = to;
            _edges[N_edges].data = e_data;
            N_edges++;
        }

        Vertex_Prop_t node_prop(uint32_t id)
        {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v)
                                     { 
                                        std::lock_guard lock(*v.mx);
                                        return v.id == id; });
            assert(node != std::end(_vertices) && "Index out of bounds");
            return node->data;
        }

        void assign(uint32_t idx, const Vertex_Prop_t &v_data)
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

        bool is_in_edge(const Edge_t &e, uint32_t idx) const
        {
            return (e.to == idx && e.from != std::numeric_limits<uint32_t>::max()) || (e.from != std::numeric_limits<uint32_t>::max() && e.to == idx);
        }

        bool is_valid(const Vertex_t &v) const
        {
            return v.id != std::numeric_limits<uint32_t>::max();
        }

        bool is_valid(const Edge_t &e) const
        {
            return e.from != std::numeric_limits<uint32_t>::max() && e.to != std::numeric_limits<uint32_t>::max();
        }

        const Vertex_t *get_neighbor(const Edge_t &e, uint32_t idx) const
        {
            const FROLS::lock_guard lock(*e.mx);
            uint32_t n_idx = 0;
            if (is_valid(e))
            {
                return e.to == idx ? get_vertex(e.from) : get_vertex(e.to);
            }
            return nullptr;
        }

        const std::vector<const Vertex_t *> neighbors(uint32_t idx) const
        {
            std::vector<const Vertex_t *> neighbors(N_vertices);
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

} // FROLS::Graph

#endif // FROLS_FROLS_GRAPH_HPP
