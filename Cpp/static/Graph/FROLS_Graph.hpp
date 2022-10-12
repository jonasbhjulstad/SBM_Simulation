//
// Created by arch on 9/29/22.
//

#ifndef FROLS_FROLS_GRAPH_HPP
#define FROLS_FROLS_GRAPH_HPP

#include <stddef.h>
#include <cassert>
#include <array>
#include <algorithm>
#include <range/v3/all.hpp>
#include <limits>

namespace FROLS::Graph {
    template<typename D>
    struct Vertex {
        int id = -1;
        D data;
    };
    template<typename D>
    struct Edge {
        D data;
        int to = -1;
        int from = -1;
    };

    template<typename V>
    class Iterator {
    public:

        using iterator_category = std::random_access_iterator_tag;
        using value_type = V;
        using difference_type = std::ptrdiff_t;
        using pointer = V *;
        using reference = V &;

    public:

        Iterator(V *ptr = nullptr) { m_ptr = ptr; }

        Iterator(const Iterator<V> &rawIterator) = default;

        ~Iterator() {}

        Iterator<V> &operator=(const Iterator<V> &rawIterator) = default;

        Iterator<V> &operator=(V *ptr) {
            m_ptr = ptr;
            return (*this);
        }

        operator bool() const {
            if (m_ptr)
                return true;
            else
                return false;
        }

        bool operator==(const Iterator<V> &rawIterator) const { return (m_ptr == rawIterator.getConstPtr()); }

        bool operator!=(const Iterator<V> &rawIterator) const { return (m_ptr != rawIterator.getConstPtr()); }

        Iterator<V> &operator+=(const difference_type &movement) {
            m_ptr += movement;
            return (*this);
        }

        Iterator<V> &operator-=(const difference_type &movement) {
            m_ptr -= movement;
            return (*this);
        }

        Iterator<V> &operator++() {
            ++m_ptr;
            return (*this);
        }

        Iterator<V> &operator--() {
            --m_ptr;
            return (*this);
        }

        Iterator<V> operator++(int) {
            auto temp(*this);
            ++m_ptr;
            return temp;
        }

        Iterator<V> operator--(int) {
            auto temp(*this);
            --m_ptr;
            return temp;
        }

        Iterator<V> operator+(const difference_type &movement) {
            auto oldPtr = m_ptr;
            m_ptr += movement;
            auto temp(*this);
            m_ptr = oldPtr;
            return temp;
        }

        Iterator<V> operator-(const difference_type &movement) {
            auto oldPtr = m_ptr;
            m_ptr -= movement;
            auto temp(*this);
            m_ptr = oldPtr;
            return temp;
        }

        difference_type operator-(const Iterator<V> &rawIterator) {
            return std::distance(rawIterator.getPtr(), this->getPtr());
        }

        V &operator*() { return *m_ptr; }

        const V &operator*() const { return *m_ptr; }

        V *operator->() { return m_ptr; }

        V *getPtr() const { return m_ptr; }

        const V *getConstPtr() const { return m_ptr; }

    protected:

        V *m_ptr;
    };


    template<typename V, typename E, size_t NV, size_t NE>
    struct GraphContainer {

        GraphContainer(const std::array<Vertex<V>, NV> vertices = std::array<Vertex<V>, NV>{},
                       const std::array<Edge<E>, NE> edges = std::array<Edge<E>, NE>{}) {
            m_data = &_vertices[0];
            _edges.end()->to = -std::numeric_limits<int>::infinity();
            _edges.end()->from = -std::numeric_limits<int>::infinity();
            std::copy(vertices.begin(), vertices.end(), _vertices.begin());
            std::copy(edges.begin(), edges.end(), _edges.begin());
        }

    public:
        struct EdgeIterator {
            EdgeIterator(E *ptr = nullptr) : m_data(ptr) {}

            typedef Iterator<E> iterator;
            typedef Iterator<const E> const_iterator;

            iterator begin() { return iterator(&m_data[0]); }

            iterator end() { return iterator(&m_data[NV]); }

            const_iterator cbegin() { return const_iterator(&m_data[0]); }

            const_iterator cend() { return const_iterator(&m_data[NV]); }

        private:
            E *m_data;
        };

        typedef Iterator<Vertex<V>> iterator;
        typedef Iterator<const Vertex<V>> const_iterator;

        iterator begin() { return iterator(&m_data[0]); }

        iterator end() {
            size_t max_idx = std::max({(int)N_vertices-1, 0});
            return iterator(&m_data[max_idx]); }

        const_iterator cbegin() { return const_iterator(&m_data[0]); }

        const_iterator cend() {
            size_t max_idx = std::max({(int)N_vertices-1, 0});
            return const_iterator(&m_data[max_idx]); }

        EdgeIterator edges;

        iterator get_vertex_iterator(size_t idx) {
            assert(idx < NV && "Index out of bounds");
            return iterator(&_vertices[idx]);
        }

        const V &get_vertex(size_t id) {
            //find vertex based on index
            auto p_V = std::find_if(_vertices.begin(), _vertices.end(), [id](const Vertex<V> &v) {
                return v.id == id;
            });
            assert(p_V != _vertices.end() && "Vertex not found");
            return p_V->data;
        }

        void assign_vertex(const V &v_data, size_t idx) {
            _vertices[idx] = v_data;
        }


        iterator get_edge_iterator(size_t index) {
            assert(index < NE && "Index out of bounds");
            return EdgeIterator(&_edges[index]);
        }

        size_t get_edge_index(iterator it) {
            return it.getPtr() - _edges.begin();
        }

        const V &get_edge(iterator it) {
            return get_edge(get_edge_index(it));
        }

        const V &get_edge(size_t idx) {
            assert(idx < NE && "Index out of bounds");
            return _vertices[idx];
        }

        void add_vertex(size_t id, const V &v_data) {
            assert(N_vertices < NV && "Max number of vertices exceeded");
            _vertices[N_vertices] = {(int)id, v_data};
            ++N_vertices;
        }


        void add_edge(size_t from, size_t to, const E e_data = {}) {
            assert(N_edges < (NE - 1) && "Exceeded max edge capacity");
            _edges[N_edges++] = {e_data, (int) from, (int) to};
        }

        void add_edge(iterator from, iterator to, const E e_data = {}) {
            add_edge(from->idx, to->idx, e_data);
        }

        V node_prop(size_t id) {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v) { return v.id == id; });
            assert(node != std::end(_vertices) && "Index out of bounds");
            return node->data;
        }


        void assign(size_t idx, const V &v_data) {
            auto vertex = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v) { return v.id == idx; });
            assert(vertex != std::end(_vertices) && "Index out of bounds");
            _vertices[idx] = Vertex<V>{(int) idx, v_data};
        }

        void remove_vertex(size_t id) {
            auto node = std::find_if(_vertices.begin(), _vertices.end(), [&](const auto &v) { return v.id == id; });
            assert(N_vertices > 0 && "No vertices to remove");
            assert(node != std::end(_vertices) && "Index not found");
            node->id = -1;
            std::for_each(_edges.begin(), _edges.end(), [&](auto &e) {
                if (e.from == id || e.to == id) {
                    e.from = -1;
                    e.to = -1;
                }
            });

            //sort vertices by id
            std::sort(_vertices.begin(), _vertices.end(), [](const auto &v1, const auto &v2) {
                return v1.id > v2.id;
            });
            N_vertices--;
        }

        void remove_edge(size_t to, size_t from) {
            auto edge = std::find_if(_edges.begin(), _edges.end(), [&](const auto &e) {
                return e.from == from && e.to == to;
            });
            assert(N_edges > 0 && "No edges to remove");
            assert(edge != std::end(_edges) && "Index not found");
            edge->from = -1;
            //sort separate edges by from
            std::sort(_edges.begin(), _edges.end(), [](const auto &e1, const auto &e2) {
                return e1.from > e2.from;
            });
            N_edges--;
        }

        void assign(iterator v, const V &v_data) {
            assign(v->idx, v_data);
        }

    protected:
        Vertex<V> *m_data;
        std::array<Vertex<V>, NV + 1> _vertices;
        std::array<Edge<E>, NE + 1> _edges;
        size_t N_vertices = 0;
        size_t N_edges = 0;
    };

    template<typename V, typename E, size_t NV, size_t NE>
    struct Graph : public GraphContainer<V, E, NV, NE> {
    public:
        Graph(const std::array<Vertex<V>, NV> vertices = {}, const std::array<Edge<E>, NE> edges = {}) : Base(vertices,
                                                                                                              edges) {}

        using Vertex_t = Vertex<V>;
        using Edge_t = Edge<E>;
        using Vertex_Prop_t = V;
        using Edge_Prop_t = E;
        static constexpr size_t MAX_VERTICES = NV;
        static constexpr size_t MAX_EDGES = NE;
        using Base = GraphContainer<V, E, NV, NE>;
        using iterator = typename Base::iterator;
        using Base::_vertices;
        using Base::_edges;

        auto neighbor_indices(Iterator<V> v) {
            return neighbor_indices(v->idx);
        }

        auto neighbor_indices(size_t idx) {
            return ranges::views::iota(10);
        }

        auto neighbors(size_t idx) {
            return Base::_edges |
                   ranges::views::filter([&](const auto &e) { return (e.to == idx) || (e.from == idx); }) |
                   ranges::views::transform([&](const auto &e) { return _vertices[e.to]; });

        }

        auto neighbors(const Iterator<V> &v) {
            return neighbors(v->idx);
        }
    };

    template<typename V, typename E, size_t NV, size_t NE>
    Graph(
            const std::array<V, NV> &vertices,
            const std::array<E, NE> &edges) -> Graph<V, E, NV, NE>;


} // FROLS::Graph

#endif //FROLS_FROLS_GRAPH_HPP
