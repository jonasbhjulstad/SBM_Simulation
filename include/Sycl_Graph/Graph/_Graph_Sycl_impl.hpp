//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_GRAPH_SYCL_HPP
#define SYCL_GRAPH_GRAPH_SYCL_HPP
#ifdef SYCL_GRAPH_USE_SYCL
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
#include <Sycl_Graph/Algorithms/Algorithms.hpp>
#include <Sycl_Graph/data_containers.hpp>

namespace Sycl_Graph::Sycl
{
  template <typename V, typename E, std::unsigned_integral uI_t>
  struct GraphContainer
  {
    // create copy constructor
    GraphContainer(GraphContainer &other) = default;
    GraphContainer(cl::sycl::queue &q, uI_t NV, uI_t NE, const cl::sycl::property_list &props = {})
        : q(q), vertex_buf(NV, props), vertex_id_buf(sycl::range<1>(NV), props), edge_buf(sycl::range<1>(NE), props), edge_to_buf(sycl::range<1>(NE), props), edge_from_buf(sycl::range<1>(NE), props), NV(NV), NE(NE)
    {
    }

    GraphContainer(cl::sycl::queue &q, const std::vector<uI_t> &vertex_ids, const std::vector<V> &vertices, const std::vector<uI_t> &edge_to_ids, const std::vector<uI_t> &edge_from_ids, const std::vector<E> &edges, const std::vector<E> &edge_props, const cl::sycl::property_list &props = {})
        : GraphContainer(q), vertex_buf(vertices, props), vertex_id_buf(vertex_ids, props), edge_buf(edges, props), edge_to_buf(edge_to_ids, props), edge_from_buf(edge_from_ids), NV(vertices.size()), NE(edges.size())
    {
      N_vertices = vertices.size();
      N_edges = edges.size();
      // compute_ev_capacities(q, N_vertices, N_edges, max_alloc);
    }
    cl::sycl::queue &q;
    cl::sycl::buffer<uI_t, 1> vertex_id_buf;
    cl::sycl::buffer<V, 1> vertex_buf;
    cl::sycl::buffer<uI_t, 1> edge_to_buf;
    cl::sycl::buffer<uI_t, 1> edge_from_buf;

    cl::sycl::buffer<E, 1> edge_buf;
    uI_t N_vertices = 0;
    uI_t N_edges = 0;
    const uI_t NV;
    const uI_t NE;
    using Vertex_t = Vertex<V, uI_t>;
    using Edge_t = Edge<E, uI_t>;
    using Vertex_Prop_t = V;
    using Edge_Prop_t = E;
    using uInt_t = uI_t;
    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();

    std::vector<V> vertex_prop(const std::vector<uI_t> &ids)
    {
      std::vector<V> vertex_props(ids.size());
      cl::sycl::buffer<V, 1> res_buf(vertex_props.data(), ids.size());
      q.submit([&](cl::sycl::handler &h)
               {
        auto out = vertex_buf.template get_access<cl::sycl::access::mode::read>(h);
        auto vertex_acc = vertex_props.template get_access<cl::sycl::access::mode::write>(h);
        h.parallel_for<class vertex_prop_search>(cl::sycl::range<1>(ids.size()), [=](cl::sycl::id<1> id)
        {
          auto it = std::find_if(vertex_acc.begin(), vertex_acc.end(), [id](const auto &v) { return v.id == id[0];});
          if (it != vertex_acc.end())
          {
            out[id] = it->data;
          }
        }); });
      return vertex_props;
    }

    uI_t find(auto condition)
    {
      uI_t idx = Vertex_t::invalid_id;
      cl::sycl::buffer<uI_t, 1> res_buf(&idx, 1);
      q.submit([&](cl::sycl::handler &h)
               {
        auto out = res_buf.template get_access<cl::sycl::access::mode::write>(h);
        auto vertex_acc = vertex_buf.template get_access<cl::sycl::access::mode::read>(h);
        find(out, vertex_acc, condition, h); });
    }

    uI_t find(auto &res_acc, auto &v_acc, auto condition, sycl::handler &h)
    {
      h.parallel_for<class vertex_id_search>(cl::sycl::range<1>(v_acc.size()), [=](cl::sycl::id<1> id)
                                             { if (condition(v_acc[id[0]])) res_acc[0] = id[0]; });
    }

    std::vector<uI_t> find_all(auto condition)
    {
      std::vector<uI_t> ids;
      cl::sycl::buffer<uI_t, 1> res_buf(ids.data(), ids.size());
      q.submit([&](cl::sycl::handler &h)
               {
        auto out = res_buf.template get_access<cl::sycl::access::mode::write>(h);
        auto vertex_acc = vertex_buf.template get_access<cl::sycl::access::mode::read>(h);
        find_all(out, vertex_acc, condition, h); });
      // remove invalid ids
      ids.erase(std::remove_if(ids.begin(), ids.end(), [](const auto &id)
                               { return id == Vertex_t::invalid_id; }),
                ids.end());
      return ids;
    }

    void find_all(auto &res_acc, auto &v_acc, auto condition, sycl::handler &h)
    {
      h.parallel_for<class vertex_id_search>(cl::sycl::range<1>(v_acc.size()), [=](cl::sycl::id<1> id)
                                             { res_acc[id[0]] = condition(v_acc[id[0]]) ? id[0] : Vertex_t::invalid_id; });
    }

    int add(const std::vector<uI_t> &ids, const std::vector<V> &v_data)
    {
      if (ids.size() + N_vertices > NV)
      {
        return false;
      }
      std::cout << "id size: " << ids.size() << ", v_data size: " << v_data.size() << ", N_vertices: " << N_vertices << std::endl;
      std::cout << "vertex_buf size: " << vertex_buf.size() << std::endl;

      cl::sycl::buffer<uI_t, 1> id_buf(ids.data(), ids.size());
      cl::sycl::buffer<V, 1> data_buf(v_data.data(), v_data.size());
      std::cout << ids.size() << ", " << v_data.size() << ", " << N_vertices << "\n"
                << std::endl;
      q.submit([&](cl::sycl::handler &h)
               {
          auto id_acc = id_buf.template get_access<cl::sycl::access::mode::read>(h);
          auto vertex_id_acc = vertex_id_buf.template get_access<cl::sycl::access::mode::write>(h);
          h.copy(id_acc, vertex_id_acc); });

      // copy data
      q.submit([&](cl::sycl::handler &h)
               {
          auto data_acc = data_buf.template get_access<cl::sycl::access::mode::read>(h);
          auto vertex_acc = vertex_buf.template get_access<cl::sycl::access::mode::write>(h);
          h.copy(data_acc, vertex_acc); });
          N_vertices += ids.size();
      return true;
    }

    bool add(const std::vector<uI_t> &from, const std::vector<uI_t> &to, const std::vector<E> &e_data = E{})
    {
      if (from.size() + N_edges > NE)
      {
        return false;
      }

      q.submit([&](cl::sycl::handler &h)
               {
        auto edge_acc = edge_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.parallel_for<class edge_add>(cl::sycl::range<1>(from.size()), [&](cl::sycl::id<1> id)
        {
          edge_acc[id[0] + N_edges].data = e_data[id[0]];
          edge_acc[id[0] + N_edges].from = from[id[0]];
          edge_acc[id[0] + N_edges].to = to[id[0]];
        }); });
      N_edges += to.size();
      return true;
    }

    bool add(const std::vector<E> &edges, const std::vector<uI_t> &from, const std::vector<uI_t> &to)
    {
      if (edges.size() + N_edges > NE)
      {
        std::cout << "Warning: too many edges to add" << std::endl;
        return false;
      }
      cl::sycl::buffer<E, 1> new_edge_buf(edges.data(), edges.size());
      cl::sycl::buffer<uI_t, 1> new_from_buf(from.data(), from.size());
      cl::sycl::buffer<uI_t, 1> new_to_buf(to.data(), to.size());
      // copy edge_buf to end of edges
      q.submit([&](cl::sycl::handler &h)
               {
        auto new_edge_acc = new_edge_buf.template get_access<cl::sycl::access::mode::read>(h);
        auto edge_acc = edge_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.copy(new_edge_acc, edge_acc); });

      q.submit([&](cl::sycl::handler &h)
               {
        auto new_to_acc = new_to_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        auto edge_to_acc = edge_to_buf.template get_access<cl::sycl::access::mode::read_write>(h);

        h.copy(new_to_acc, edge_to_acc); });
      q.submit([&](cl::sycl::handler &h)
               {
        auto new_from_acc = new_from_buf.template get_access<cl::sycl::access::mode::read>(h);
        auto edge_from_acc = edge_from_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.copy(new_from_acc, edge_from_acc); });
      N_edges += edges.size();

      return true;
    }

    void assign(const std::vector<uI_t> &id, const std::vector<V> &v_data)
    {
      q.submit([&](cl::sycl::handler &h)
               {
        auto vertex_acc = vertex_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.parallel_for<class vertex_assign>(cl::sycl::range<1>(id.size()), [=](cl::sycl::id<1> id)
        {
          auto it = std::find_if(vertex_acc.begin(), vertex_acc.end(), [id](const auto &v) { return v.id == id[0];});
          if (it != vertex_acc.end())
          {
            it->data = v_data[id[0]];
          }
        }); });
    }

    void remove(const std::vector<uI_t> &id)
    {
      q.submit([&](cl::sycl::handler &h)
               {
        auto vertex_acc = vertex_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.parallel_for<class vertex_remove>(cl::sycl::range<1>(id.size()), [=](cl::sycl::id<1> id)
        {
          auto it = std::find_if(vertex_acc.begin(), vertex_acc.end(), [id](const auto &v) { return v.id == id[0];});
          if (it != vertex_acc.end())
          {
            it->id = Vertex_t::invalid_id;
          }
        }); });
      N_vertices -= id.size();
    }

    void sort_vertices()
    {
      q.submit([&](cl::sycl::handler &h)
               {
        auto vertex_acc = vertex_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        //sort vertices in vertex_acc by id
        h.single_task([=]()
        {
          std::sort(vertex_acc.begin(), vertex_acc.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
        }); });
    }

    void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from)
    {
      // Set ids of edges to invalid_id
      q.submit([&](cl::sycl::handler &h)
               {
        auto edge_acc = edge_buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.parallel_for<class edge_remove>(cl::sycl::range<1>(to.size()), [=](cl::sycl::id<1> id)
        {
          auto it = std::find_if(edge_acc.begin(), edge_acc.end(), [id, from, to](const auto &e) { return e.from == from[id[0]] && e.to == to[id[0]];});
          if (it != edge_acc.end())
          {
            it->from = Edge_t::invalid_id;
            it->to = Edge_t::invalid_id;
          }
        }); });
      N_edges -= to.size();
    }
  };
  template <typename V, typename E, std::unsigned_integral uI_t>
  struct Graph
  {
    using Container_t = GraphContainer<V, E, uI_t>;
    using Vertex_Prop_t = V;
    using Edge_Prop_t = E;
    using Vertex_t = typename Container_t::Vertex_t;
    using Edge_t = typename Container_t::Edge_t;
    using uInt_t = uI_t;
    Graph(cl::sycl::queue &q, uI_t NV, uI_t NE, const cl::sycl::property_list &props = {}) : C(q, NV, NE, props) {}
    Graph(cl::sycl::queue &q, const std::vector<Vertex<V, uI_t>> &vertices,
          const std::vector<Edge<E, uI_t>> &edges, const cl::sycl::property_list &props = {}) : C(q, vertices, edges, props) {}
    Container_t C;
    static constexpr auto invalid_id = Container_t::invalid_id;
    // Graph(GraphContainer<V, E, uI_t, NV, NE> &container) : container(container) {}
    auto vertex_access(cl::sycl::handler &h) { return C.vertex_buf.template get_access<cl::sycl::access::mode::read_write>(h); }
    auto edge_to_access(cl::sycl::handler &h) { return C.edge_to_buf.template get_access<cl::sycl::access::mode::read_write>(h); }
    auto edge_from_access(cl::sycl::handler &h) { return C.edge_from_buf.template get_access<cl::sycl::access::mode::read_write>(h); }
    auto edge_access(cl::sycl::handler &h) { return C.edge_buf.template get_access<cl::sycl::access::mode::read_write>(h); }
    uI_t &N_vertices = C.N_vertices;
    uI_t &N_edges = C.N_edges;
    const uI_t &NV = C.NV;
    const uI_t &NE = C.NE;

    std::vector<uI_t> get_N_out_edges(const std::vector<uI_t> &ids, auto condition)
    {

      std::vector<uI_t> N_neighbors(ids.size());
      cl::sycl::buffer<uI_t, 1> res_buf(N_neighbors.data(), ids.size());
      C.q.submit([&](cl::sycl::handler &h)
                 {
        auto out = res_buf.template get_access<cl::sycl::access::mode::write>(h);
        auto vertex_acc = vertex_access(h);
        h.parallel_for<class vertex_id_search>(cl::sycl::range<1>(ids.size()), [&, this](cl::sycl::id<1> id)
        {
          for (int i = 0; i < N_vertices; i++)
          {
            if ((ids[id[0]] == vertex_acc[i].id) && condition(vertex_acc[i]))
            {
              out[id[0]]++;
            }
          }
        }); });
      return N_neighbors;
    }

    void edge_sort_out()
    {
      edge_sort([](const Edge_t &edge)
                { return edge.src < edge.dst; });
    }

    void edge_sort(auto condition)
    {
      C.q.submit([&](cl::sycl::handler &h)
                 {
        auto edge_acc = edge_access(h);
        Sycl_Graph::Sycl::algorithms::bitonic_sort(edge_acc, h, condition); });
    }

    uI_t get_neighbor_id(const Edge_t &edge, uI_t id)
    {
      return edge.src == id ? edge.dst : edge.src;
    }

    Vertex_t find(auto vertex_acc, uI_t id)
    {
      return C.find(vertex_acc, id);
    }

    // Forward-declarations

    std::vector<uI_t>
    find(auto condition)
    {
      return C.find(condition);
    }
    bool add(const std::vector<E> &edges, const std::vector<uI_t> &to_ids, const std::vector<uI_t> &from_ids)
    {
      return C.add(edges, to_ids, from_ids);
    }

    int add(const std::vector<uI_t> &ids, const std::vector<V> &v_data)
    {
      return C.add(ids, v_data);
    }

    bool add(const std::vector<uI_t> &from, const std::vector<uI_t> &to, const std::vector<E> &e_data = E{})
    {
      return C.add(from, to, e_data);
    }

    void assign(const std::vector<uI_t> &id, const std::vector<V> &v_data)
    {
      return C.assign(id, v_data);
    }

    void remove(const std::vector<uI_t> &id)
    {
      return C.remove(id);
    }

    void sort_vertices()
    {
      C.sort_vertices();
    }

    void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from)
    {
      return C.remove(to, from);
    }
  };
}

#endif
#endif // Sycl_Graph_hpp
