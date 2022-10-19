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
        std::mutex* mx;
    };
    template <typename D>
    struct Edge
    {
        D data;
        uint32_t to = std::numeric_limits<uint32_t>::max();
        uint32_t from = std::numeric_limits<uint32_t>::max();
        std::mutex* mx;
    };


    template <typename Container>
    struct Graph;

    template <typename V, typename E, uint32_t NV, uint32_t NE>
    struct ArrayGraphContainer
    {
        ArrayGraphContainer(std::array<std::mutex*, NV + 1> &v_mx, std::array<std::mutex*, NE + 1> &e_mx) : NV_max(NV), NE_max(NE)
        {
            m_data = &_vertices[0];
            for (int i = 0; i < NV_max + 1; i++)
            {
                _vertices[i].mx = v_mx[i];
            }

            for (int i = 0; i < NE_max + 1; i++)
            {
                _edges[i].mx = e_mx[i];
            }
        }
        const uint32_t NV_max;
        const uint32_t NE_max;

        template <typename T>
        using VertexContainer = std::array<T, NV + 1>;
        template <typename T>
        using EdgeContainer = std::array<T, NE + 1>;


        protected : Vertex<V> *m_data;
        std::array<Vertex<V>, NV + 1> _vertices;
        std::array<Edge<E>, NE + 1> _edges;
        uint32_t N_vertices = 0;
        uint32_t N_edges = 0;
    };

    template <typename V, typename E>
    struct VectorGraphContainer
    {
        VectorGraphContainer(std::vector<std::mutex*> &v_mx, std::vector<std::mutex*> &e_mx) : NV_max(v_mx.size()), NE_max(e_mx.size())
        {
            m_data = &_vertices[0];
            _vertices.resize(NV_max);
            _edges.resize(NE_max);
            for (int i = 0; i < NV_max + 1; i++)
            {
                _vertices[i].mx = v_mx[i];
            }

            for (int i = 0; i < NE_max + 1; i++)
            {
                _edges[i].mx = e_mx[i];
            }
        }
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        using Vertex_t = Vertex<V>;
        using Edge_t = Edge<E>;
        template <typename T>
        struct VertexContainer: public std::vector<T>
        {
            VertexContainer() : std::vector<T>(NV_max) {}
        };
        template <typename T>
        using EdgeContainer = std::vector<T>;

        const uint32_t NV_max;
        const uint32_t NE_max;

        Vertex<V> *m_data;
        std::vector<Vertex<V>> _vertices;
        std::vector<Edge<E>> _edges;
        uint32_t N_vertices = 0;
        uint32_t N_edges = 0;
    };

    template <typename Container>
    struct Graph
    {
        using Vertex_t = typename Container::Vertex_t;
        using Edge_t =  typename Container::Edge_t;
        using Vertex_Prop_t = typename Container::Vertex_Prop_t;
        using Edge_Prop_t = typename Container::Edge_Prop_t;
        template <typename T>
        using VertexContainer = typename Container::template VertexContainer<T>;
        template <typename T>
        using EdgeContainer = typename Container::template EdgeContainer<T>;

        using V = typename Container::Vertex_Prop_t;
        using E = typename Container::Edge_Prop_t;
        const uint32_t& NV_max = container.NV_max;
        const uint32_t& NE_max = container.NE_max;

        Graph(auto& v_mx, auto& e_mx) : container(v_mx, e_mx) {}

        // member typedefs provided through inheriting from std::iterator
        class iterator: public std::iterator<
                            std::input_iterator_tag,   // iterator_category
                            uint16_t,                      // value_type
                            uint16_t,                      // difference_type
                            const uint16_t*,               // pointer
                            uint16_t                       // reference
                                        >{
            uint16_t num = 0;
            uint16_t N_max = 0;
            VertexContainer<Vertex_t>& _vertices;

        public:
            explicit iterator(uint16_t _num, uint16_t max, auto& vertices) : num(_num), N_max(max), _vertices(vertices) {}
            iterator& operator++() {num = N_max >= 0 ? num + 1: num - 1; return *this;}
            iterator operator++(int) {iterator retval = *this; ++(*this); return retval;}
            bool operator==(iterator other) const {return num == other.num;}
            bool operator!=(iterator other) const {return !(*this == other);}
            Vertex_t operator*() const {return _vertices[num];}
        };
        iterator begin() {return iterator(0, N_vertices, container._vertices);}
        iterator end() {return iterator(N_vertices, N_vertices, container._vertices);}


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
            assert(container._vertices[0].id == 0);
            // find vertex based on index
            auto p_V = std::find_if(container._vertices.begin(), container._vertices.end(), [id](Vertex<V> v)
                                    { return v.id == id; });
            assert(p_V != container._vertices.end() && "Vertex not found");
            return (Vertex<V>*)&(*p_V);
        }

        void assign_vertex(const V &v_data, uint32_t idx)
        {
            // find index of vertex
            auto p_V = std::find_if(container._vertices.begin(), container._vertices.end(), [idx](const Vertex<V> &v)
                                    {
                std::lock_guard lock(*v.mx);
                return v.id == idx; });
            assert(p_V != container._vertices.end() && "Vertex not found");
            p_V->data = v_data;
        }

        void add_vertex(uint32_t id, const V &v_data)
        {
            assert(N_vertices < NV_max && "Max number of vertices exceeded");
            std::lock_guard lock(*container._vertices[N_vertices].mx);
            container._vertices[N_vertices].id = id;
            container._vertices[N_vertices].data = v_data;
            ++N_vertices;
        }

        void add_edge(uint32_t from, uint32_t to, const E e_data = {})
        {
            assert(N_edges < NE_max && "Max number of edges exceeded");
            std::lock_guard lock(*_edges[N_edges].mx);
            _edges[N_edges].from = from;
            _edges[N_edges].to = to;
            _edges[N_edges].data = e_data;
            N_edges++;
        }

        V node_prop(uint32_t id)
        {
            auto node = std::find_if(container._vertices.begin(), container._vertices.end(), [&](const auto &v)
                                     { 
                                        std::lock_guard lock(*v.mx);
                                        return v.id == id; });
            assert(node != std::end(container._vertices) && "Index out of bounds");
            return node->data;
        }

        void assign(uint32_t idx, const V &v_data)
        {
            auto vertex = std::find_if(container._vertices.begin(), container._vertices.end(), [&](const auto &v)
                                       { 
                std::lock_guard lock(*v.mx);
                return v.id == idx; });
            assert(vertex != std::end(container._vertices) && "Index out of bounds");
            vertex->data = v_data;
        }

        void remove_vertex(uint32_t id)
        {
            auto node = std::find_if(container._vertices.begin(), container._vertices.end(), [&](const auto &v)
                                     { return v.id == id; });
            assert(N_vertices > 0 && "No vertices to remove");
            assert(node != std::end(container._vertices) && "Index not found");
            node->id = std::numeric_limits<uint32_t>::max();
            std::for_each(_edges.begin(), _edges.end(), [&](auto &e)
                          {
                std::lock_guard lock(*e.mx);
                if (e.from == id || e.to == id) {
                    e.from = std::numeric_limits<uint32_t>::max();
                    e.to = std::numeric_limits<uint32_t>::max();
                } });

            // sort vertices by id
            std::sort(container._vertices.begin(), container._vertices.end(), [](const auto &v1, const auto &v2)
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

        const VertexContainer<const Vertex<V> *> neighbors(uint32_t idx) const
        {
            VertexContainer<const Vertex<V> *> neighbors;
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

        protected:
        Container container;
        EdgeContainer<Edge_t>& _edges = container._edges;
        uint32_t& N_vertices = container.N_vertices;
        uint32_t& N_edges = container.N_edges;
    };

    template <typename V, typename E, uint32_t NV, uint32_t NE>
    using ArrayGraph = Graph<ArrayGraphContainer<V,E, NV, NE>>;
    template <typename V, typename E>
    using VectorGraph = Graph<VectorGraphContainer<V, E>>;

    


} // FROLS::Graph

#endif // FROLS_FROLS_GRAPH_HPP
