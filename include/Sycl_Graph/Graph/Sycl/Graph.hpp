//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_SYCL_GRAPH_HPP
#define SYCL_GRAPH_SYCL_GRAPH_HPP
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <type_traits>
#include <utility>
// #include <Sycl_Graph/execution.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <type_traits>
namespace Sycl_Graph::Sycl
{

  template <typename V, typename E, std::unsigned_integral uI_t = uint32_t, std::floating_point dType = float>
  struct Graph : public ::Sycl_Graph::Graph_Base<V, E, Vertex_Buffer<V, uI_t>, Edge_Buffer<E, uI_t>, uI_t, dType>

  {
    typedef ::Sycl_Graph::Graph_Base<V, E, Vertex_Buffer<V, uI_t>, Edge_Buffer<E, uI_t>, uI_t, dType> Base_t;
    // create copy constructor
    Graph(sycl::queue &q, uI_t NV, uI_t NE, const sycl::property_list &props = {})
        : q(q), vertex_buf(q, NV, props), edge_buf(q, NE, props), Base_t(vertex_buf, edge_buf) {}

    Graph(sycl::queue &q, const std::vector<Vertex<V, uI_t>> &vertices,
          const std::vector<Edge<E, uI_t>> &edges = {},
          const sycl::property_list &props = {})
        : vertex_buf(q, vertices, props), edge_buf(q, edges, props), q(q), Base_t(vertex_buf, edge_buf) {}
    sycl::queue &q;
    Vertex_Buffer<V, uI_t> vertex_buf;
    Edge_Buffer<E, uI_t> edge_buf;
    uI_t Graph_ID = 0;
    //convert to typedef
    typedef Vertex<V, uI_t> Vertex_t;
    typedef Edge<E, uI_t> Edge_t;
    typedef V Vertex_Prop_t;
    typedef E Edge_Prop_t;
    typedef uI_t uInt_t;
    typedef Graph<V, E, uI_t> Graph_t;  

    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();
    uI_t N_vertices() const
    {
      return vertex_buf.size();
    }
    uI_t N_edges() const
    {
      return edge_buf.size();
    }

    uI_t max_vertices() const
    {
      return vertex_buf.NV;
    }

    uI_t max_edges() const
    {
      return edge_buf.NE;
    }

    void resize(uI_t NV_new, uI_t NE_new)
    {
      vertex_buf.resize(NV_new);
      edge_buf.resize(NE_new);
    }

    Graph_t operator+(const Graph_t &other) const
    {
      //call derived copy constructor
      Graph_t result(*this);
      result.vertex_buf += other.vertex_buf;
      result.edge_buf += other.edge_buf;
      return result;
    }


    Graph_t operator=(Graph_t other)
    {
      vertex_buf = other.vertex_buf;
      edge_buf = other.edge_buf;
      return *this;
    }

    Graph_t &operator=(Graph_t &other)
    {
      vertex_buf = other.vertex_buf;
      edge_buf = other.edge_buf;
      return *this;
    }

    Graph_t &operator+=(const Graph_t &other)
    {
      vertex_buf += other.vertex_buf;
      edge_buf += other.edge_buf;
      return *this;
    }

  // find vertex index based on condition
  template <typename T> uI_t find(T condition) {
    uI_t idx = Vertex_t::invalid_id;
    sycl::buffer<uI_t, 1> res_buf(&idx, 1);
    q.submit([&](sycl::handler &h) {
      auto out = res_buf.template get_access<sycl::access::mode::write>(h);
      auto vertex_acc =
          vertex_buf.template get_access<sycl::access::mode::read>(h);
      find(out, vertex_acc, condition, h);
    });
    q.wait();
    return idx;
  }

  template <typename T0, typename T1, typename T2>
  void find(T0 &res_acc, T1 &v_acc, T2 condition, sycl::handler &h) {
    h.parallel_for<class vertex_id_search>(sycl::range<1>(v_acc.size()),
                                           [=](sycl::id<1> id) {
                                             if (condition(v_acc[id[0]]))
                                               res_acc[0] = id[0];
                                           });
  }

    template <sycl::access::mode mode>
    auto get_vertex_access(sycl::handler &h)
    {
      return vertex_buf.template get_access<mode>(h);
    }

    template <sycl::access::mode mode>
    auto get_edge_access(sycl::handler &h)
    {
      return edge_buf.template get_access<mode>(h);
    }
  };
} // namespace Sycl_Graph::Sycl

#endif
