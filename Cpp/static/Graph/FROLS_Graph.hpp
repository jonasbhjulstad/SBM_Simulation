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
// #include <range/v3/all.hpp>
#include <limits>

namespace FROLS::Graph {
    template<typename D>
    struct Vertex {
        uint16_t id = std::numeric_limits<uint16_t>::max();
        D data;
    };
    template<typename D>
    struct Edge {
        D data;
        uint16_t to = std::numeric_limits<uint16_t>::max();
        uint16_t from = std::numeric_limits<uint16_t>::max();
    };

    template<typename V, typename E, uint16_t NV, uint16_t NE>
    struct GraphContainer {

        GraphContainer(const std::array<Vertex<V>, NV> vertices = std::array<Vertex<V>, NV>{},
                       const std::array<Edge<E>, NE> edges = std::array<Edge<E>, NE>{}) {
            m_data = &_vertices[0];
            _edges.end()->to = std::numeric_limits<uint16_t>::max();
            _edges.end()->from = std::numeric_limits<uint16_t>::max();
            std::copy(vertices.begin(), vertices.end(), _vertices.begin());
            std::copy(edges.begin(), edges.end(), _edges.begin());
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

        const V &get_vertex(uint16_t id) {
            //find vertex based on index
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [id](const Vertex<V> &v) {
                return v.id == id;
            });
            assert(p_V != _vertices.end() && "Vertex not found");
            return p_V->data;
        }

        void assign_vertex(const V &v_data, uint16_t idx) {
            _vertices[idx] = v_data;
        }



        void add_vertex(uint16_t id, const V &v_data) {
            assert(N_vertices < NV && "Max number of vertices exceeded");
            _vertices[N_vertices] = {id, v_data};
            ++N_vertices;
        }


        void add_edge(uint16_t from, uint16_t to, const E e_data = {}) {
            assert(N_edges < (NE - 1) && "Exceeded max edge capacity");
            _edges[N_edges++] = {e_data, from,  to};
        }


        V node_prop(uint16_t id) {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v) { return v.id == id; });
            assert(node != std::end(_vertices) && "Index out of bounds");
            return node->data;
        }


        void assign(uint16_t idx, const V &v_data) {
            auto vertex = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v) { return v.id == idx; });
            assert(vertex != std::end(_vertices) && "Index out of bounds");
            _vertices[idx] = Vertex<V>{idx, v_data};
        }

        void remove_vertex(uint16_t id) {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v) { return v.id == id; });
            assert(N_vertices > 0 && "No vertices to remove");
            assert(node != std::end(_vertices) && "Index not found");
            node->id = std::numeric_limits<uint16_t>::max();
            std::for_each(_edges.begin(), _edges.end(), [&](auto &e) {
                if (e.from == id || e.to == id) {
                    e.from = std::numeric_limits<uint16_t>::max();
                    e.to = std::numeric_limits<uint16_t>::max();
                }
            });

            //sort vertices by id
            std::sort(_vertices.begin(), _vertices.end(), [](const auto &v1, const auto &v2) {
                return v1.id > v2.id;
            });
            N_vertices--;
        }

        void remove_edge(uint16_t to, uint16_t from) {
            auto edge = std::find_if(_edges.begin(), _edges.end(), [&](const auto &e) {
                return e.from == from && e.to == to;
            });
            assert(N_edges > 0 && "No edges to remove");
            assert(edge != std::end(_edges) && "Index not found");
            edge->from = std::numeric_limits<uint16_t>::max();
            //sort separate edges by from
            std::sort(_edges.begin(), _edges.end(), [](const auto &e1, const auto &e2) {
                return e1.from > e2.from;
            });
            N_edges--;
        }
        bool is_in_edge(const Edge<E>& e, uint16_t idx)
        {
            return (e.from == idx) || (e.to == idx);
        }


    protected:
        Vertex<V> *m_data;
        std::array<Vertex<V>, NV + 1> _vertices;
        std::array<Edge<E>, NE + 1> _edges;
        uint16_t N_vertices = 0;
        uint16_t N_edges = 0;
    };

    template<typename V, typename E, uint16_t NV, uint16_t NE>
    struct Graph : public GraphContainer<V, E, NV, NE> {
    public:
        Graph(const std::array<Vertex<V>, NV> vertices = {}, const std::array<Edge<E>, NE> edges = {}) : Base(vertices,
                                                                                                              edges) {}
        std::array<Vertex<V>, NV> tmp_vertices;
        using Vertex_t = Vertex<V>;
        using Edge_t = Edge<E>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        static constexpr uint16_t MAX_VERTICES = NV;
        static constexpr uint16_t MAX_EDGES = NE;
        using Base = GraphContainer<V, E, NV, NE>;
        using Base::_vertices;
        using Base::_edges;

        auto neighbors(uint16_t idx) {
            // auto rang = Base::_edges |
            //        ranges::views::filter([&](const auto &e) { return (e.to == idx) || (e.from == idx); }) |
            //        ranges::views::transform([&](const auto &e) { return _vertices[e.to]; });
            // std::cout << std::distance(rang.begin(), rang.end()) << std::endl;
            // return Base::_edges |
            //        ranges::views::filter([&](const auto &e) { return (e.to == idx) || (e.from == idx); }) |
            //        ranges::views::transform([&](const auto &e) { return _vertices[e.to]; });
            std::copy_if(_vertices.begin(), _vertices.end(), tmp_vertices.begin(), [&](const auto &v) {
                return std::find_if(_edges.begin(), _edges.end(), [&](const auto &e) {
                    return (e.to == idx && e.from != std::numeric_limits<uint16_t>::max()) || (e.from !=std::numeric_limits<uint16_t>::max() && e.to == idx);
                }) != _edges.end();
            });
            return tmp_vertices;
        }
    };

    template<typename V, typename E, uint16_t NV, uint16_t NE>
    Graph(
            const std::array<V, NV> &vertices,
            const std::array<E, NE> &edges) -> Graph<V, E, NV, NE>;

    


} // FROLS::Graph

#endif //FROLS_FROLS_GRAPH_HPP
