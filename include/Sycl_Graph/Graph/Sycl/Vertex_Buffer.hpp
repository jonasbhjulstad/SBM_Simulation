#ifndef SYCL_GRAPH_GRAPH_VERTEX_HPP
#define SYCL_GRAPH_GRAPH_VERTEX_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/buffer_routines.hpp>
namespace Sycl_Graph::Sycl {
template <typename V, std::unsigned_integral uI_t, sycl::access::mode mode>
struct Vertex_Accessor {
  Vertex_Accessor(sycl::buffer<V, 1> &v_buf, sycl::buffer<uI_t, 1> &id_buf,
                  sycl::handler &h, sycl::property_list props = {})
      : data(v_buf, h, props), id(id_buf, h, props) {}
  uI_t size() const { return data.size(); }
  sycl::accessor<V, 1, mode> data;
  sycl::accessor<uI_t, 1, mode> id;
};

enum Vertex_Indexing { VERTEX_INDEX_ID, VERTEX_INDEX_POSITION };


template <typename V, std::unsigned_integral uI_t> struct Vertex_Buffer : public Vertex_Buffer_Base<V, uI_t, Vertex_Buffer<V, uI_t>>{
static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();

  uI_t N_vertices = 0;
  sycl::buffer<uI_t, 1> id_buf;
  sycl::buffer<V, 1> data_buf;
  sycl::queue &q;


  Vertex_Buffer(sycl::queue &q, uI_t NV, const sycl::property_list &props = {})
      : id_buf(sycl::range<1>(NV), props), data_buf(sycl::range<1>(NV), props), q(q) {}

  Vertex_Buffer(sycl::queue &q, const std::vector<Vertex<V, uI_t>> &vertices,
                const sycl::property_list &props = {})
      : id_buf(sycl::range<1>(vertices.size()), props),
        data_buf(sycl::range<1>(vertices.size()), props),
        q(q) {
    std::vector<uI_t> ids(vertices.size());
    std::vector<V> v_data(vertices.size());
    for (uI_t i = 0; i < vertices.size(); ++i) {
      ids[i] = vertices[i].id;
      v_data[i] = vertices[i].data;
    }
    add(ids, v_data);
  }
  uI_t size() const { return data_buf.size(); };
  template <sycl::access::mode mode>
  Vertex_Accessor<V, uI_t, mode> get_access(sycl::handler &h) {
    return Vertex_Accessor<V, uI_t, mode>(data_buf, id_buf, h);
  }

  void resize(uI_t new_size) 
  {
    N_vertices = (N_vertices > new_size) ? N_vertices : new_size;
    buffer_resize(q, id_buf, new_size);
    buffer_resize(q, data_buf, new_size);
  }

  void add(const std::vector<uI_t> &ids, const std::vector<V> &v_data = {}) {
    std::vector<V> data = (v_data.size() == 0) ? std::vector<V>(ids.size()) : v_data;
    if(N_vertices + ids.size() > data_buf.size())
    {
      resize(N_vertices + ids.size());
    }
    host_buffer_add(data_buf, data, q);
    host_buffer_add(id_buf, ids, q);
    N_vertices += ids.size();
  }

  std::vector<uI_t> get_valid_ids() {
    std::vector<uI_t> ids;
    ids.reserve(N_vertices);
    auto id_acc = id_buf.template get_access<sycl::access::mode::read>();
    for (uI_t i = 0; i < id_buf.size(); ++i) {
      if (id_acc[i] != invalid_id) {
        ids.push_back(id_acc[i]);
      }
    }
    return ids;
  }

  std::vector<V> get_data(const std::vector<uI_t> &ids) {
    std::vector<V> result(ids.size());

    sycl::buffer<V, 1> res_buf(result.data(), ids.size());
    q.submit([&](sycl::handler &h) {
      auto out = res_buf.template get_access<sycl::access::mode::read>(h);
      auto v_acc = data_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for<class vertex_prop_search>(
          sycl::range<1>(ids.size()), [=](sycl::id<1> id) {
            auto it =
                std::find_if(v_acc.begin(), v_acc.end(),
                             [id](const auto &v) { return v.id == id[0]; });
            if (it != v_acc.end()) {
              out[id] = it->data;
            }
          });
    });
    return result;
  }

  std::vector<Vertex<V, uI_t>> get_vertices() {
    std::vector<Vertex<V, uI_t>> result(N_vertices);
    auto id_acc = id_buf.template get_access<sycl::access::mode::read>();
    auto v_acc = data_buf.template get_access<sycl::access::mode::read>();
    uI_t i = 0;
    for (uI_t j = 0; j < id_buf.size(); ++j) {
      if (id_acc[j] != invalid_id) {
        result[i].id = id_acc[j];
        result[i].data = v_acc[j];
        ++i;
      }
    }
    return result;
  }

  template <Vertex_Indexing idx_type = VERTEX_INDEX_ID>
  void assign(const std::vector<uI_t> &id, const std::vector<V> &data) {
    q.submit([&](sycl::handler &h) {
      auto v_acc =
          data_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for<class vertex_assign>(
          sycl::range<1>(id.size()), [=](sycl::id<1> id) {
            auto it =
                std::find_if(v_acc.begin(), v_acc.end(),
                             [id](const auto &v) { return v.id == id[0]; });
            if (it != v_acc.end()) {
              it->data = data[id[0]];
            }
          });
    });
  }

  void remove(const std::vector<uI_t> &id) {
    q.submit([&](sycl::handler &h) {
      auto v_acc =
          data_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for<class vertex_remove>(
          sycl::range<1>(id.size()), [=](sycl::id<1> id) {
            auto it =
                std::find_if(v_acc.begin(), v_acc.end(),
                             [id](const auto &v) { return v.id == id[0]; });
            if (it != v_acc.end()) {
              it->id = invalid_id;
            }
          });
    });
    N_vertices -= id.size();
  }

  Vertex_Buffer<V, uI_t> &operator=(const Vertex_Buffer<V, uI_t> &other) {
    id_buf = other.id_buf;
    data_buf = other.data_buf;
    N_vertices = other.N_vertices;
    return *this;
  }

  Vertex_Buffer<V, uI_t> operator+(const Vertex_Buffer<V, uI_t> &other) {
    

    id_buf = device_buffer_combine(q, id_buf, other.id_buf, this->N_vertices,
                                   other.N_vertices);
    data_buf = device_buffer_combine(q, data_buf, other.data_buf,
                                     this->N_vertices, other.N_vertices);
    N_vertices += other.N_vertices;
    return *this;
  }


  size_t byte_size() { return data_buf.get_size() + id_buf.get_size(); }
};


} // namespace Sycl_Graph::Sycl

#endif