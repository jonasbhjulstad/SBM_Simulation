#ifndef SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#define SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
namespace Sycl_Graph::Sycl {
template <Sycl_Graph::Base::Edge_type Edge_t, sycl::access::mode Mode>
struct Edge_Accessor {
  typedef typename Edge_t::uI_t uI_t;

  Edge_Accessor(sycl::buffer<Edge_t, 1> &edge_buf, sycl::buffer<uI_t, 1> &to_buf,
                sycl::buffer<uI_t, 1> &from_buf, sycl::handler &h,
                sycl::property_list props = {})
      : data(edge_buf, h, props), to(to_buf, h, props),
        from(from_buf, h, props) {}
  sycl::accessor<Edge_t, 1, Mode> data;
  sycl::accessor<uI_t, 1, Mode> to;
  sycl::accessor<uI_t, 1, Mode> from;
};

template <Sycl_Graph::Base::Edge_type _Edge_t> 
struct Edge_Buffer: public Sycl_Graph::Base::Edge_Buffer<_Edge_t, Edge_Buffer<_Edge_t>>
 {
  typedef Sycl_Graph::Base::Edge_Buffer<_Edge_t, Edge_Buffer<_Edge_t>> Base_t;
  typedef typename Base_t::Container_t Container_t;
  typedef _Edge_t Edge_t;
  typedef typename Edge_t::uI_t uI_t;
  typedef typename Edge_t::Data_t Data_t;

  Edge_Buffer(sycl::queue &q, uI_t NE, const sycl::property_list &props = {})
      : 

  Edge_Buffer(sycl::queue &q, const std::vector<Edge_t> &edges,
              const sycl::property_list &props = {})
      : to_buf(sycl::range<1>(edges.size()), props),
        from_buf(sycl::range<1>(edges.size()), props),
        data_buf(sycl::range<1>(edges.size()), props), NE(edges.size()), q(q){


  std::vector<Edge_ID_Pair<uI_t>> get_valid_ids() {
    std::vector<Edge_ID_Pair<uI_t>> id_pairs;
    id_pairs.reserve(N_edges);
    q.submit([&](sycl::handler &h) {
      auto to_acc = to_buf.template get_access<sycl::access::mode::read>(h);
      auto from_acc =
          from_buf.template get_access<sycl::access::mode::read>(h);
      h.parallel_for<class edge_get_valid_ids>(
          sycl::range<1>(N_edges), [=](sycl::id<1> id) {
            if (to_acc[id] != invalid_id && from_acc[id] != invalid_id) {
              id_pairs.push_back({to_acc[id], from_acc[id]});
            }
          });
    });
    return id_pairs;
  }

  std::vector<Edge_t> get_edges()
  {
      std::vector<Edge_t> edges;
      edges.reserve(N_edges);
      auto to_acc = to_buf.template get_access<sycl::access::mode::read>();
      auto from_acc = from_buf.template get_access<sycl::access::mode::read>();
      auto data_acc = data_buf.template get_access<sycl::access::mode::read>();
      for (uI_t i = 0; i < N_edges; ++i)
      {
          if (to_acc[i] != invalid_id && from_acc[i] != invalid_id)
          {
              edges.push_back({data_acc[i], to_acc[i], from_acc[i]});
          }
      }
      return edges;
  }

  void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from) {
    // Set ids of edges to invalid_id
    q.submit([&](sycl::handler &h) {
      auto data_acc =
          data_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for<class edge_remove>(
          sycl::range<1>(to.size()), [=](sycl::id<1> id) {
            auto it = std::find_if(data_acc.begin(), data_acc.end(),
                                   [id, from, to](const auto &e) {
                                     return e.from == from[id[0]] &&
                                            e.to == to[id[0]];
                                   });
            if (it != data_acc.end()) {
              it->from = invalid_id;
              it->to = invalid_id;
            }
          });
    });
    N_edges -= to.size();
  }

  Edge_Buffer<Edge_t> &operator=(const Edge_Buffer<Edge_t> &other) {
    to_buf = other.to_buf;
    from_buf = other.from_buf;
    data_buf = other.data_buf;
    N_edges = other.N_edges;
    return *this;
  }

  Edge_Buffer<Edge_t> operator+(const Edge_Buffer<Edge_t> &other) {
    to_buf = device_buffer_combine(q, to_buf, other.to_buf, this->N_edges,
                                   other.N_edges);
    from_buf = device_buffer_combine(q, from_buf, other.from_buf, this->N_edges,
                                     other.N_edges);
    data_buf = device_buffer_combine(q, data_buf, other.data_buf, this->N_edges,
                                     other.N_edges);
    N_edges = to_buf.size();
    return *this;
  }
  size_t byte_size() {
    return to_buf.size() * sizeof(uI_t) +
           from_buf.size() * sizeof(uI_t) +
           data_buf.size() * sizeof(Data_t);
  }
};

template <typename T>
concept Edge_Buffer_type = Sycl_Graph::Base::Edge_Buffer_type<T>;
} // namespace sycl_graph

#endif // 