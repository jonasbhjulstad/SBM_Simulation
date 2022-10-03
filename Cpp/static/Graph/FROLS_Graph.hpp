//
// Created by arch on 9/29/22.
//

#ifndef FROLS_FROLS_GRAPH_HPP
#define FROLS_FROLS_GRAPH_HPP

#include <stddef.h>
#include <array>
#include <ranges>

namespace FROLS::Graph {
    template<typename D>
    struct Vertex {
        D data;
    };
    template<typename D>
    struct Edge {
        D data;
        size_t to, from;
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
        GraphContainer(const std::array<V, NV> &vertices, const std::array<E, NE> &edges) : _vertices(vertices),
                                                                                            _edges(edges),
                                                                                            m_data(_vertices.data()){

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

        typedef Iterator<V> iterator;
        typedef Iterator<const V> const_iterator;

        iterator begin() { return iterator(&m_data[0]); }

        iterator end() { return iterator(&m_data[NV]); }

        const_iterator cbegin() { return const_iterator(&m_data[0]); }

        const_iterator cend() { return const_iterator(&m_data[NV]); }

        EdgeIterator edges;

        iterator get_vertex_iterator(size_t index) {
            return iterator(&(_vertices.data() + index));
        }

        size_t get_vertex_index(iterator it) {
            return it.getPtr() - _vertices.begin();
        }

        const V &get_vertex(iterator it) {
            return get_vertex(get_vertex_index(it));
        }

        const V &get_vertex(size_t idx) {
            return _vertices[idx];
        }

        void assign_vertex(const V &v_data, size_t idx) {
            _vertices[idx] = v_data;
        }


        iterator get_edge_iterator(size_t index) {
            return iterator(&(_vertices.data() + index));
        }

        size_t get_edge_index(iterator it) {
            return it.getPtr() - _vertices.begin();
        }

        const V &get_edge(iterator it) {
            return get_edge(get_edge_index(it));
        }

        const V &get_edge(size_t idx) {
            return _vertices[idx];
        }


    protected:


        V *m_data;
        std::array<V, NV> _vertices;
        std::array<E, NE> _edges;
    };

    template<typename V, typename E, size_t NV, size_t NE>
    struct Graph : public GraphContainer<V, E, NV, NE> {
    public:
        Graph(const std::array<V, NV> &vertices, const std::array<E, NE> &edges) : Base(vertices, edges) {}
        using Base = GraphContainer<V, E, NV, NE>;
        using iterator = typename Base::iterator;
        using Base::_vertices;


        std::views::_Filter get_neighbors(iterator iv) {
            auto is_in_edge = [&](const E &edge) {
                return (this->get_vertex_index(iv) == edge.to) || (this->get_vertex_index(iv) == edge.from);
            };
            return std::views::all(_vertices) | std::ranges::views::filter(is_in_edge) |
                   std::views::transform([&](const V &vertex) { return iterator(vertex); });
        }

    };

    template<typename V, typename E, size_t NV, size_t NE>
    Graph(const std::array<V, NV> &vertices, const std::array<E, NE> &edges) -> Graph<V, E, NV, NE>;


} // FROLS::Graph

#endif //FROLS_FROLS_GRAPH_HPP
