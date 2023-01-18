//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_GRAPH_SYCL_HPP
#define SYCL_GRAPH_GRAPH_SYCL_HPP
#include <algorithm>
#include <numeric>
#include <array>
#include <cassert>
#include <limits>
#include <mutex>
#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <iterator>
// #include <Sycl_Graph/execution.hpp>
#include <type_traits>
#include <CL/sycl.hpp>
#include "Graph_Types.hpp"
#include "Graph_Types_Sycl.hpp"

namespace Sycl_Graph::Sycl
{
  
  template <typename V, typename E, typename uI_t>
  struct Graph
  {
    // create copy constructor
    Graph(Graph &other) = default;
    Graph(sycl::queue &q, uI_t NV, uI_t NE, const sycl::property_list &props = {})
        : q(q), vertex_buf(NV, q, props), edge_buf(NE, q, props)
    {
    }

    Graph(sycl::queue &q, const std::vector<Vertex<V, uI_t>>& vertices, const std::vector<Edge<E, uI_t>>& edges, const sycl::property_list &props = {})
        : Graph(q), vertex_buf(vertices, props), edge_buf(edges, props)
    {
    }
    sycl::queue &q;
    Vertex_Buffer<V, uI_t> vertex_buf;
    Edge_Buffer<E, uI_t> edge_buf;
    const uI_t& NV = vertex_buf.NV;
    const uI_t& NE = edge_buf.NE;
    using Vertex_t = Vertex<V, uI_t>;
    using Edge_t = Edge<E, uI_t>;
    using Vertex_Prop_t = V;
    using Edge_Prop_t = E;
    using uInt_t = uI_t;
    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();
    inline uI_t N_vertices() const {return vertex_buf.N_vertices;}
    inline uI_t N_edges() const {return edge_buf.N_edges;}


    //find vertex index based on condition
    template <typename T>
    uI_t find(T condition)
    {
      uI_t idx = Vertex_t::invalid_id;
      sycl::buffer<uI_t, 1> res_buf(&idx, 1);
      q.submit([&](sycl::handler &h)
               {
        auto out = res_buf.template get_access<sycl::access::mode::write>(h);
        auto vertex_acc = vertex_buf.template get_access<sycl::access::mode::read>(h);
        find(out, vertex_acc, condition, h); });
      q.wait();
      return idx;
    }

    template <typename T0, typename T1, typename T2>
    void find(T0 &res_acc, T1 &v_acc, T2 condition, sycl::handler &h)
    {
      h.parallel_for<class vertex_id_search>(sycl::range<1>(v_acc.size()), [=](sycl::id<1> id)
                                             { if (condition(v_acc[id[0]])) res_acc[0] = id[0]; });

    }


    //perfect forward methods of buffers

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

    template <typename... Args>
    void add_vertex(Args &&... args)
    {
      vertex_buf.add(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void add_edge(Args &&... args)
    {
      edge_buf.add(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void remove_vertex(Args &&... args)
    {
      vertex_buf.remove(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void remove_edge(Args &&... args)
    {
      edge_buf.remove(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void assign_vertex(Args &&... args)
    {
      vertex_buf.assign(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void assign_edge(Args &&... args)
    {
      edge_buf.assign(std::forward<Args>(args)...);
    }

    template <typename... Args>
    V get_vertex(Args &&... args)
    {
      return vertex_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    E get_edge(Args &&... args)
    {
      return edge_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<V> get_vertex_data(Args &&... args)
    {
      return vertex_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<E> get_edge_data(Args &&... args)
    {
      return edge_buf.get_data(std::forward<Args>(args)...);
    }

  };
}

#endif
