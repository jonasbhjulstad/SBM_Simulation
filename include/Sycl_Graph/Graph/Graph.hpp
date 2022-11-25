//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_FROLS_GRAPH_HPP
#define SYCL_GRAPH_FROLS_GRAPH_HPP

#include <Sycl_Graph/data_containers.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <iterator>

namespace Sycl_Graph {

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
struct GraphContainerBase {
  using Vertex_t = Vertex<V, uI_t>;
  using Edge_t = Edge<E, uI_t>;
  auto begin() {
    const auto &derived = static_cast<Derived const &>(*this);
    return derived.v_begin();
  }

  auto vertex_iterator(uI_t idx) {
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
  auto edge_iterator(uI_t idx) {
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

  V vertex_prop(uI_t id) {
    const auto &derived = static_cast<Derived const &>(*this);

    auto it = std::find_if(derived.vertices.begin(), derived.vertices.end(),
                           [id](const auto &v) { return v.id == id; });
    if (it != derived.vertices.end()) {
      return it->data;
    }
    return {};
  }

  auto find(uI_t id) const {
    const auto &derived = static_cast<Derived const &>(*this);

    return std::find_if(derived.vertices.begin(), derived.vertices.end(),
                        [id](const auto &v) { return v.id == id; });
  }

  inline const Vertex_t *get_neighbor(const Edge_t &e, uI_t id) const {
    return (e.from == id) ? &(*find(e.to)) : &(*find(e.from));
  }

  void add(uI_t id, const V &v_data) {
    auto &derived = static_cast<Derived &>(*this);
    derived.vertices[N_vertices].id = id;
    derived.vertices[N_vertices].data = v_data;
    N_vertices++;
  }


  void add(uI_t from, uI_t to, const E &e_data = E{}) {
    auto &derived = static_cast<Derived&>(*this);
    derived.edges[N_edges].from = from;
    derived.edges[N_edges].to = to;
    derived.edges[N_edges].data = e_data;
    N_edges++;
  }

  void assign(uI_t id, const V &v_data) {
    auto& derived = static_cast<Derived&>(*this);
    auto v = find(id);
    derived.vertices[std::distance(derived.vertices.cbegin(), v)].data = v_data;
  }

  void remove(uI_t id) {
    assert(N_vertices > 0 && "No vertices to remove");
    const auto &derived = static_cast<Derived const &>(*this);
    auto v = find(id);
    assert(v != derived.v_end() && "Index not found");
    v->id = std::numeric_limits<uI_t>::max();
    N_vertices--;
  }

  void sort_vertices() {
    const auto &derived = static_cast<Derived const &>(*this);
    std::sort(derived.vertices.begin(), derived.vertices.end(),
              [](const auto &v1, const auto &v2) { return v1.id < v2.id; });
  }

  void remove_edge(uI_t to, uI_t from) {
    const auto &derived = static_cast<Derived const &>(*this);

    auto edge = std::find_if(
        derived._edges.begin(), derived._edges.end(),
        [&](const auto &e) { return e.from == from && e.to == to; });
    assert(N_edges > 0 && "No edges to remove");
    assert(edge != std::end(derived._edges) && "Index not found");
    edge->from = std::numeric_limits<uI_t>::max();
    // sort separate edges by from
    std::sort(derived._edges.begin(), derived._edges.end(),
              [](const auto &e1, const auto &e2) {
                std::lock_guard lock1(*e1.mx);
                std::lock_guard lock2(*e2.mx);
                return e1.from < e2.from;
              });
    N_edges--;
  }

  uI_t N_vertices = 0;
  uI_t N_edges = 0;
};
namespace Fixed {
template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE,
          template <typename, uI_t> typename Array_t>
struct GraphContainer
    : public GraphContainerBase<GraphContainer<V, E, uI_t, NV, NE, Array_t>,
                                uI_t, V, E> {
  using Base = GraphContainerBase<GraphContainer<V, E, uI_t, NV, NE, Array_t>,
                                  uI_t, V, E>;
  Array_t<Vertex<V, uI_t>, NV> vertices;
  Array_t<Edge<E, uI_t>, NE> edges;
  uI_t &N_vertices = Base::N_vertices;
  uI_t &N_edges = Base::N_edges;
  auto begin() { return std::begin(vertices); }
  auto end() { return std::begin(vertices) + NV; }

  void assign(uI_t id, const V &v_data) {
    auto v = std::find_if(std::begin(vertices), std::end(vertices),
                          [id](const auto &v) { return v.id == id; });
    assert(v != std::end(vertices) && "Index not found");
    v->data = v_data;
  }
};

template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE,
          std::unsigned_integral uA_t,
          template <typename, uA_t> typename Array_t>
struct Graph : public GraphContainer<V, E, uI_t, NV, NE, Array_t> {
  static constexpr uI_t NV_MAX = NV;
  static constexpr uI_t NE_MAX = NE;

  using Vertex_t = Vertex<V, uI_t>;
  using Edge_t = Edge<E, uI_t>;
  using Vertex_Prop_t = V;
  using Edge_Prop_t = E;
  using Container_t = GraphContainer<V, E, uI_t, NV, NE, Array_t>;
  Container_t C;
  uI_t N_vertices = 0;
  uI_t N_edges = 0;

  const Vertex_Prop_t &operator[](uI_t id) const { return get_vertex_prop(id); }

  const Vertex_Prop_t &get_vertex_prop(uI_t id) const {
    return get_vertex(id)->data;
  }

  const Array_t<Vertex_t *, NV + 1> neighbors(uI_t idx) const {
    Array_t<Vertex_t *, NV + 1> neighbors = {};
    std::for_each(C._edges.begin(), C._edges.end(),
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
} // namespace Fixed

namespace Dynamic {
template <typename V, typename E, std::unsigned_integral uI_t,
          template <typename> typename Array_t>
struct GraphContainer
    : public GraphContainerBase<GraphContainer<V, E, uI_t, Array_t>, uI_t, V,
                                E> {
  using Base =
      GraphContainerBase<GraphContainer<V, E, uI_t, Array_t>, uI_t, V, E>;
  GraphContainer(uI_t NV_max, uI_t NE_max)
      : NV_max(NV_max), NE_max(NE_max), vertices(NV_max), edges(NE_max) {}
  Array_t<Vertex<V, uI_t>> vertices;
  Array_t<Edge<E, uI_t>> edges;
  uI_t &N_vertices = Base::N_vertices;
  uI_t &N_edges = Base::N_edges;
  auto begin() { return std::begin(vertices); }
  auto end() { return std::begin(vertices) + N_vertices; }
  uI_t NV_max, NE_max;
};

template <typename V, typename E, std::unsigned_integral uI_t,
          template <typename> typename Array_t>
struct Graph : public GraphContainer<V, E, uI_t, Array_t> {
  Graph(uI_t NV_max, uI_t NE_max)
      : GraphContainer<V, E, uI_t, Array_t>(NV_max, NE_max) {}
  using Vertex_t = Vertex<V, uI_t>;
  using Edge_t = Edge<E, uI_t>;
  using Vertex_Prop_t = V;
  using Edge_Prop_t = E;
  using Base = GraphContainer<V, E, uI_t, Array_t>;
  using Base::edges;
  uI_t& N_vertices = Base::N_vertices;
  uI_t& N_edges = Base::N_vertices;
  using Base::vertices;

  const Vertex_Prop_t &operator[](uI_t id) const { return get_vertex_prop(id); }

  const Vertex_Prop_t &get_vertex_prop(uI_t id) const {
    return get_vertex(id)->data;
  }

  inline bool is_edge_valid(const Edge_t &e) const {
    return e.from != std::numeric_limits<uI_t>::max() &&
           e.to != std::numeric_limits<uI_t>::max();
  }

  inline bool is_in_edge(const Edge_t &e, uI_t idx) const {

    return is_edge_valid(e) && (e.from == idx || e.to == idx);
  }

  const Array_t<const Vertex_t *> neighbors(uI_t idx) const {
    Array_t<const Vertex_t *> neighbors(N_vertices);
    std::for_each(edges.begin(), edges.end(), [&, N = 0](const auto e) mutable {
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
} // namespace Dynamic

} // namespace Sycl_Graph

#endif // FROLS_FROLS_GRAPH_HPP
