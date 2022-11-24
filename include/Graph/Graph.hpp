//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_FROLS_GRAPH_HPP
#define SYCL_GRAPH_FROLS_GRAPH_HPP

#include <Data_Containers.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>
#include <utility>

namespace Sycl::Graph {

template <typename D, std::unsigned_integral ID_t> struct Vertex {
  ID_t id = std::numeric_limits<ID_t>::max();
  D data;
};
template <typename D, std::unsigned_integral ID_t> struct Edge {
  D data;
  ID_t to = std::numeric_limits<ID_t>::max();
  ID_t from = std::numeric_limits<ID_t>::max();
};

template <typename Derived, std::unsigned_integral uI_t, typename V, typename E>
struct GraphContainer {
  GraphContainer(uI_t NV_max, uI_t NE_max)
      : NV_max(NV_max), NE_max(NE_max){}

  auto begin() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.v_begin();
  }

  auto vertex_iterator(uI_t idx)
  {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.vertex_iterator(idx);
  }

  auto end() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.v_end();
  }

  auto e_begin() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.e_begin();
  }
  auto edge_iterator(uI_t idx)
  {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.edge_iterator(idx);
  }
  auto e_end() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.e_end();
  }

  uI_t get_max_vertices() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.get_max_vertices();
  }

  uI_t get_max_edges() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.get_max_edges();
  }

  uI_t get_vertex_mutex_count() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.get_vertex_mutex_count();
  }

  uI_t get_edge_mutex_count() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.get_edge_mutex_count();
  }

  V node_prop(uI_t id) {
    const auto &derived = static_cast<Derived const &>(*this);

        auto it = std::find_if(derived._vertices.begin(), derived._vertices.end(), [id](const auto &v) { return v.id == id; });
        if (it != derived._vertices.end()) {
          return it->data;
        }
    return {};
  }

  auto find(uI_t id)
  {
    const auto &derived = static_cast<Derived const &>(*this);

    for (int i = 0; i < NV_max; i += NV_per_mx)
    {
        auto start = derived.vertex_iterator(i);
        auto end = (i + NV_per_mx) < NV_max ? derived.vertex_iterator(i + NV_per_mx) : derived.v_end();
        auto it = std::find_if(start, end, [id](const auto &v) { return v.id == id; });
        if (it != end) {
          return it;
        }
    }
  }

  void assign(uI_t id, const V &v_data) {
    const auto &derived = static_cast<Derived const &>(*this);

    auto v = find(id);
    v->id = id;
    v->data = v_data;
  }

  void remove(uI_t id) {
    assert(N_vertices > 0 && "No vertices to remove");
    const auto &derived = static_cast<Derived const &>(*this);
    auto v = find(id);
    assert(v != derived.v_end() && "Index not found");
    v->id = std::numeric_limits<uI_t>::max();
    N_vertices--;
  }

  void sort_vertices()
  {
    const auto &derived = static_cast<Derived const &>(*this);
    for (int i = 0; i < NV_max; i += NV_per_mx)
    {
        auto start = derived.vertex_iterator(i);
        auto end = (i + NV_per_mx) < NV_max ? derived.vertex_iterator(i + NV_per_mx) : derived.v_end();
         std::sort(start, end,
              [](const auto &v1, const auto &v2) {
                return v1.id < v2.id;
              });   
    }
  }

  void remove_edge(uI_t to, uI_t from) {
    const auto &derived = static_cast<Derived const &>(*this);

    auto edge = std::find_if(derived._edges.begin(), derived._edges.end(), [&](const auto &e) {
      return e.from == from && e.to == to;
    });
    assert(N_edges > 0 && "No edges to remove");
    assert(edge != std::end(derived._edges) && "Index not found");
    edge->from = std::numeric_limits<uI_t>::max();
    // sort separate edges by from
    std::sort(derived._edges.begin(), derived._edges.end(), [](const auto &e1, const auto &e2) {
      std::lock_guard lock1(*e1.mx);
      std::lock_guard lock2(*e2.mx);
      return e1.from < e2.from;
    });
    N_edges--;
  }

  uI_t N_vertices = 0;
  uI_t N_edges = 0;
  const uI_t NV_max, NE_max, NV_mx, NE_mx, NV_per_mx, NE_per_mx;
};

template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE,
          uI_t NV_MX, uI_t NE_MX,
          template <typename, uI_t> typename FixedArray_t>
struct FixedGraphContainer
    : public GraphContainer<
          FixedGraphContainer<V, E, uI_t, NV, NE, NV_MX, NE_MX, FixedArray_t>,
          uI_t, V, E> {
  using Base = GraphContainer<
      FixedGraphContainer<V, E, uI_t, NV, NE, NV_MX, NE_MX, FixedArray_t>, uI_t,
      V, E>;
  FixedArray_t<V, NV> vertices;
  FixedArray_t<E, NE> edges;
  uI_t &N_vertices = Base::N_vertices;
  uI_t &N_edges = Base::N_edges;
  auto begin() { return std::begin(vertices); }
  auto end() { return std::begin(vertices) + NV; }

  void add(uI_t id, const V &v_data) {
    vertices[N_vertices].id = id;
    vertices[N_vertices].data = v_data;
    N_vertices++;
  }

  void add(uI_t from, uI_t to, const E &e_data) {
    edges[N_edges].from = from;
    edges[N_edges].to = to;
    edges[N_edges].data = e_data;
    N_edges++;
  }
};

template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE,
          template <class...> typename Array_t>
requires containers::GenericArray<Array_t<V>>
struct Graph {
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


  const Vertex_Prop_t &operator[](uI_t id) const { return get_vertex_prop(id); }

  const Vertex_Prop_t &get_vertex_prop(uI_t id) const {
    return get_vertex(id)->data;
  }

  bool is_in_edge(const Edge_t &e, uI_t idx) const {
    return (e.to == idx && e.from != std::numeric_limits<uI_t>::max()) ||
           (e.from != std::numeric_limits<uI_t>::max() && e.to == idx);
  }

  bool is_valid(const Vertex_t &v) const {
    return v.id != std::numeric_limits<uI_t>::max();
  }

  bool is_valid(const Edge_t &e) const {
    return e.from != std::numeric_limits<uI_t>::max() &&
           e.to != std::numeric_limits<uI_t>::max();
  }

  const Vertex_t *get_neighbor(const Edge_t &e, uI_t idx) const {
    if (is_valid(e)) {
      return e.to == idx ? get_vertex(e.from) : get_vertex(e.to);
    }
    return nullptr;
  }

  const std::array<Vertex_t *, NV + 1> neighbors(uI_t idx) const {
    std::array<Vertex_t *, NV + 1> neighbors = {};
    if (neighbors.size() < NV_max) {
      neighbors.resize(NV_max);
    }
    std::for_each(_edges.begin(), _edges.end(),
                  [&, N = 0](const auto e) mutable {
                    if (is_in_edge(e, idx)) {
                      const auto p_nv = this->get_neighbor(e, idx);
                      if (p_nv != nullptr) {
                        neighbors[N] = p_nv;
                        N++;
                      }
                    }
                  });
    return neighbors;
  }
};

} // namespace Sycl::Graph

#endif // FROLS_FROLS_GRAPH_HPP
