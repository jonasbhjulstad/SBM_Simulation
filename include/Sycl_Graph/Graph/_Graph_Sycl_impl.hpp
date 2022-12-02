//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_GRAPH_SYCL_HPP
#define SYCL_GRAPH_GRAPH_SYCL_HPP
#ifdef SYCL_GRAPH_USE_SYCL
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
// #include <Sycl_Graph/execution.hpp>
#include <type_traits>
#include <CL/sycl.hpp>
#include "Graph_Types.hpp"

namespace Sycl_Graph::Sycl
  {
    template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE>
    struct GraphContainer
    {
    GraphContainer(sycl::queue &q) : q(q), 
    {
    }
    GraphContainer(sycl::queue &q, const std::vector<Vertex<V, uI_t>> &vertices,
                   const std::vector<Edge<E, uI_t>> &edges, const sycl::property_list &props = {})
        : GraphContainer(q), vertex_buf(vertices, props), edge_buf(edges, props)
    {
      N_vertices = vertices.size();
      N_edges = edges.size();
      // compute_ev_capacities(q, N_vertices, N_edges, max_alloc);
    }
    sycl::queue &q;
    sycl::buffer<Vertex<V, uI_t>, 1> vertex_buf;
    sycl::buffer<Edge<E, uI_t>, 1> edge_buf;
    sycl::buffer<Vertex<V, uI_t>, 1> scatter_buf;
    uI_t N_vertices;
    uI_t N_edges;
    using Vertex_t = Vertex<V, uI_t>;
    using Edge_t = Edge<E, uI_t>;
    using Vertex_Prop_t = V;
    using Edge_Prop_t = E;

    uI_t get_max_vertices()
    {
      return NV;
    }

    uI_t get_max_edges()
    {
      return NE;
    }

    std::vector<V> vertex_prop(const std::vector<uI_t> &ids)
    {
      std::vector<V> vertex_props(ids.size());
      sycl::buffer<V, 1> res_buf(vertex_props.data(), ids.size());
      q.submit([&](sycl::handler& h)
      {
        auto out = vertex_buf.template get_access<sycl::access::mode::read>(h);
        auto vertex_acc = vertex_props.template get_access<sycl::access::mode::write>(h);
        h.parallel_for<class vertex_prop_search>(sycl::range<1>(ids.size()), [=](sycl::id<1> id)
        {
          auto it = std::find_if(vertex_acc.begin(), vertex_acc.end(), [id](const auto &v) { return v.id == id[0];});
          if (it != vertex_acc.end())
          {
            out[id] = it->data;
          }
        });
      });
      return vertex_props;
    }

    int add(const std::vector<uI_t> &ids, const std::vector<V> &v_data)
    {
        if (ids.size() + N_vertices > NV)
        {
          return false;
        }

        q.submit([&](sycl::handler& h)
        {
          auto vertex_acc = vertex_buf.template get_access<sycl::access::mode::read_write>(h);
          h.parallel_for<class vertex_add>(sycl::range<1>(ids.size()), [=](sycl::id<1> id)
          {
            vertex_acc[id[0] + N_vertices] = Vertex_t{ids[id[0]], v_data[id[0]]};
          });
        });
        N_vertices += ids.size();
        return true;
    }

    bool add(const std::vector<uI_t> &from, const std::vector<uI_t> &to, const std::vector<E> &e_data = E{})
    {
      if (from.size() + N_edges > NE)
      {
        return false;
      }

      q.submit([&](sycl::handler& h)
      {
        auto edge_acc = edge_buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for<class edge_add>(sycl::range<1>(from.size()), [=](sycl::id<1> id)
        {
          edge_acc[id[0] + N_edges] = Edge_t{from[id[0]], to[id[0]], e_data[id[0]]};
        });
      });
      N_edges += to.size();
      return true;
    }

    void assign(const std::vector<uI_t> &id, const std::vector<V> &v_data)
    {
      q.submit([&](sycl::handler& h)
      {
        auto vertex_acc = vertex_buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for<class vertex_assign>(sycl::range<1>(id.size()), [=](sycl::id<1> id)
        {
          auto it = std::find_if(vertex_acc.begin(), vertex_acc.end(), [id](const auto &v) { return v.id == id[0];});
          if (it != vertex_acc.end())
          {
            it->data = v_data[id[0]];
          }
        });
      });
    }

    void remove(const std::vector<uI_t> &id)
    {
      q.submit([&](sycl::handler& h)
      {
        auto vertex_acc = vertex_buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for<class vertex_remove>(sycl::range<1>(id.size()), [=](sycl::id<1> id)
        {
          auto it = std::find_if(vertex_acc.begin(), vertex_acc.end(), [id](const auto &v) { return v.id == id[0];});
          if (it != vertex_acc.end())
          {
            it->id = Vertex_t::invalid_id;
          }
        });
      });
      N_vertices -= id.size();
    }

    void sort_vertices()
    {
      q.submit([&](sycl::handler& h)
      {
        auto vertex_acc = vertex_buf.template get_access<sycl::access::mode::read_write>(h);
        //sort vertices in vertex_acc by id
        h.single_task([=]()
        {
          std::sort(vertex_acc.begin(), vertex_acc.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
        });
      });
    }

    void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from)
    {
      //Set ids of edges to invalid_id
      q.submit([&](sycl::handler& h)
      {
        auto edge_acc = edge_buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for<class edge_remove>(sycl::range<1>(to.size()), [=](sycl::id<1> id)
        {
          auto it = std::find_if(edge_acc.begin(), edge_acc.end(), [id, from, to](const auto &e) { return e.from == from[id[0]] && e.to == to[id[0]];});
          if (it != edge_acc.end())
          {
            it->from = Edge_t::invalid_id;
            it->to = Edge_t::invalid_id;
          }
        });
      });
      N_edges -= to.size();
    }

  private:
    // void compute_ev_capacities(sycl::queue &q, uI_t N_vertices, uI_t N_edges, size_t max_alloc)
    // {
    //   float ev_ratio = N_vertices / N_edges;

    //   // get maximum allocatable memory
    //   auto device = q.get_device();
    //   auto max_device_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    //   max_alloc = std::min(max_alloc, max_device_alloc);

    //   // auto curr_alloc = sizeof(Vertex<V, uI_t>) * N_vertices + sizeof(Edge<E, uI_t>) * N_edges;
      
    //   // auto max_allocatable = max_alloc - curr_alloc;
    //   // NV_max = std::min({((uI_t)ev_ratio * max_allocatable / sizeof(Vertex<V, uI_t>) + N_vertices), NV_max});
    //   // NE_max = std::min({(uI_t)((1 - ev_ratio) * max_allocatable / sizeof(Edge<E, uI_t>) + N_edges), NE_max});
    // }

  };
  template <typename V, typename E, std::unsigned_integral uI_t, uI_t NV, uI_t NE>
  struct Graph
  {
    using Container_t = GraphContainer<V, E, uI_t, NV, NE>;
    using Vertex_t = typename Container_t::Vertex_t;
    using Edge_t = typename Container_t::Edge_t;
    using Vertex_Prop_t = typename Container_t::Vertex_Prop_t;
    using Edge_Prop_t = typename Container_t::Edge_Prop_t;
    Graph(sycl::queue& q): C(q){}
    Graph(sycl::queue& q, const std::vector<Vertex<V, uI_t>> &vertices,
          const std::vector<Edge<E, uI_t>> &edges, const sycl::property_list &props = {}) : C(q, vertices, edges, props) {}
    // Graph(GraphContainer<V, E, uI_t, NV, NE> &container) : container(container) {}

  private:
    Container_t C;
  };
}


#endif
#endif // Sycl_Graph_hpp
