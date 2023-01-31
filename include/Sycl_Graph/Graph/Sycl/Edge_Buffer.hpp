#ifndef SYCL_GRAPH_EDGE_HPP
#define SYCL_GRAPH_EDGE_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/buffer_routines.hpp>
namespace Sycl_Graph::Sycl {
template <typename E, typename uI_t, sycl::access::mode Mode>
struct Edge_Accessor {
  Edge_Accessor(sycl::buffer<E, 1> &edge_buf, sycl::buffer<uI_t, 1> &to_buf,
                sycl::buffer<uI_t, 1> &from_buf, sycl::handler &h,
                sycl::property_list props = {})
      : data(edge_buf, h, props), to(to_buf, h, props),
        from(from_buf, h, props) {}
  sycl::accessor<E, 1, Mode> data;
  sycl::accessor<uI_t, 1, Mode> to;
  sycl::accessor<uI_t, 1, Mode> from;
};

enum Edge_Indexing { EDGE_INDEXING_ID, EDGE_INDEXING_POSITION };
template <typename V, typename uI_t> struct Vertex_Buffer;

template <typename uI_t = uint32_t>
struct Edge_ID_Pair
{
    static constexpr uI_t invalid_ID = std::numeric_limits<uI_t>::max();
    Edge_ID_Pair() = default;
    bool valid() const
    {
        return to != invalid_ID && from != invalid_ID;
    }
    uI_t to = invalid_ID;
    uI_t from = invalid_ID;

};

template <typename E, typename uI_t> struct Edge_Buffer: public Edge_Buffer_Base<E, uI_t, Edge_Buffer<E, uI_t>>
 {
  // current number of edges
  uI_t N_edges = 0;
  // maximum number of edges
  const uI_t NE;
  sycl::queue &q;
  sycl::buffer<uI_t, 1> to_buf;
  sycl::buffer<uI_t, 1> from_buf;
  sycl::buffer<E, 1> data_buf;

  Edge_Indexing index_type = EDGE_INDEXING_ID;

  static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
  Edge_Buffer(sycl::queue &q, uI_t NE, const sycl::property_list &props = {})
      : to_buf(sycl::range<1>(NE), props), from_buf(sycl::range<1>(NE), props),
        data_buf(sycl::range<1>(NE), props), NE(NE), q(q) {}

  Edge_Buffer(sycl::queue &q, const std::vector<Edge<E, uI_t>> &edges,
              const sycl::property_list &props = {})
      : to_buf(sycl::range<1>(edges.size()), props),
        from_buf(sycl::range<1>(edges.size()), props),
        data_buf(sycl::range<1>(edges.size()), props), NE(edges.size()), q(q){
    // split edges into to/from and data
    std::vector<uI_t> edge_to(edges.size());
    std::vector<uI_t> edge_from(edges.size());
    std::vector<E> edge_data(edges.size());
    for (uI_t i = 0; i < edges.size(); ++i) {
      edge_to[i] = edges[i].to;
      edge_from[i] = edges[i].from;
      edge_data[i] = edges[i].data;
    }
      host_buffer_copy(q, to_buf, edge_to);
      host_buffer_copy(q, from_buf, edge_from);
      host_buffer_copy(q, data_buf, edge_data);
  }

  uI_t size() const { return N_edges; }
  template <sycl::access::mode Mode>
  Edge_Accessor<E, uI_t, Mode> get_access(sycl::handler &h) {
    return Edge_Accessor<E, uI_t, Mode>(data_buf, to_buf, from_buf, h);
  }
  void resize(uI_t new_size) {
    N_edges = (N_edges > new_size) ? N_edges : new_size;
    buffer_resize(to_buf, new_size, q);
    buffer_resize(from_buf, new_size, q);
    buffer_resize(data_buf, new_size, q);
  }

  void add(const std::vector<uI_t> &to, const std::vector<uI_t> &from,
           const std::vector<E> &data, uI_t offset = 0) {
    host_buffer_add(to_buf, to, q, offset);
    host_buffer_add(from_buf, from, q, offset);
    host_buffer_add(data_buf, data, q, offset);

    N_edges += to.size();
  }

  void add(const sycl::buffer<uI_t, 1> &to, const sycl::buffer<uI_t, 1> &from,
           const sycl::buffer<E, 1> &data, uI_t offset = 0) {
    host_buffer_add(to_buf, to, q, N_edges, offset);
    host_buffer_add(from_buf, from, q, N_edges, offset);
    host_buffer_add(data_buf, data, q, N_edges, offset);
    N_edges += to.get_count();
  }

  std::vector<Edge_ID_Pair<uI_t>> get_valid_edge_ids() {
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

  std::vector<Edge<E, uI_t>> get_edges()
  {
      std::vector<Edge<E, uI_t>> edges;
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

  template <typename V>
  void positional_index_convert(Vertex_Buffer<V, uI_t> &v_buf) {
    if (index_type == EDGE_INDEXING_POSITION) {
      return;
    }
    uI_t N_vertices = v_buf.size();
    uI_t N_edges = this->size();
    q.submit([&](sycl::handler &h) {
      auto v_acc = v_buf.template get_access<sycl::access::mode::read>(h);
      auto to_acc =
          to_buf.template get_access<sycl::access::mode::read_write>(h);
      auto from_acc =
          from_buf.template get_access<sycl::access::mode::read_write>(h);

      h.parallel_for(sycl::range<1>(N_edges), [=](sycl::id<1> id) {
        for (int i = 0; i < N_vertices; i++) {
          if (to_acc[id] == v_acc.id[i]) {
            to_acc[id] = i;
          }
          if (from_acc[id] == v_acc.id[i]) {
            from_acc[id] = i;
          }
        }
      });
    });
    index_type = EDGE_INDEXING_POSITION;
  }

  Edge_Buffer<E, uI_t> &operator=(const Edge_Buffer<E, uI_t> &other) {
    to_buf = other.to_buf;
    from_buf = other.from_buf;
    data_buf = other.data_buf;
    N_edges = other.N_edges;
    return *this;
  }

  Edge_Buffer<E, uI_t> operator+(const Edge_Buffer<E, uI_t> &other) {
    to_buf = device_buffer_combine(q, to_buf, other.to_buf, this->N_edges,
                                   other.N_edges);
    from_buf = device_buffer_combine(q, from_buf, other.from_buf, this->N_edges,
                                     other.N_edges);
    data_buf = device_buffer_combine(q, data_buf, other.data_buf, this->N_edges,
                                     other.N_edges);
    N_edges = to_buf.size();
    return *this;
  }

  template <typename V> void id_index_convert(Vertex_Buffer<V, uI_t> &v_buf) {
    if (index_type == EDGE_INDEXING_ID) {
      return;
    }
    q.submit([&](sycl::handler &h) {
      auto v_acc = v_buf.get_access<sycl::access::mode::read>(h);
      auto to_acc =
          to_buf.template get_access<sycl::access::mode::read_write>(h);
      auto from_acc =
          from_buf.template get_access<sycl::access::mode::read_write>(h);

      h.parallel_for(sycl::range<1>(to_acc.size()), [&](sycl::id<1> id) {
        to_acc[id] = v_acc[to_acc[id]].id;
        from_acc[id] = v_acc[from_acc[id]].id;
      });
    });
    index_type = EDGE_INDEXING_ID;
  }
  size_t byte_size() {
    return to_buf.get_count() * sizeof(uI_t) +
           from_buf.get_count() * sizeof(uI_t) +
           data_buf.get_count() * sizeof(E);
  }
};
} // namespace sycl_graph

#endif // 