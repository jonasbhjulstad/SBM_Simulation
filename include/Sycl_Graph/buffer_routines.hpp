#ifndef SYCL_GRAPH_BUFFER_ROUTINES_HPP
#define SYCL_GRAPH_BUFFER_ROUTINES_HPP
#include <CL/sycl.hpp>
#include <string>
#include <vector>
namespace Sycl_Graph {
template <typename T>
inline void host_buffer_copy(sycl::queue& q, sycl::buffer<T, 1> &buf, const std::vector<T> &vec) 
{
  if(vec.size() == 0) {
    return;
  }
  sycl::buffer<T, 1> tmp_buf(vec.data(), sycl::range<1>(vec.size()));
  q.submit([&](sycl::handler &h) {
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    auto tmp_acc = tmp_buf.template get_access<sycl::access::mode::read>(h);
    h.parallel_for(vec.size(), [=](sycl::id<1> i) { acc[i] = tmp_acc[i]; });
  });
}

template <typename T> class host_buffer_copy_kernel;

template <typename T, std::unsigned_integer uI_t = uint32_t>
void host_buffer_add(sycl::buffer<T, 1> &buf, const std::vector<T> &vec,
                     sycl::queue &q, uI_t offset = 0) {
  if (vec.size() == 0) {
    return;
  }
  if constexpr (sizeof(T) > 0) {
    sycl::buffer<T, 1> tmp_buf(vec.data(), sycl::range<1>(vec.size()));
    q.submit([&](sycl::handler &h) {
      auto tmp_acc = tmp_buf.template get_access<sycl::access::mode::read>(h);
      auto acc = buf.template get_access<sycl::access::mode::write>(h);

      h.parallel_for<host_buffer_copy_kernel<T>>(
          vec.size(), [=](sycl::id<1> i) { acc[i + offset] = tmp_acc[i]; });
    });
    // submit with kernel_name
  }
}

template <typename T, std::unsigned_integer uI_t = uint32_t>
void device_buffer_add(sycl::buffer<T, 1> &dest_buf, sycl::buffer<T, 1> src_buf,
                       sycl::queue &q, uI_t offset = 0) {
  if (src_buf.get_count() == 0) {
    return;
  }
  if constexpr (sizeof(T) > 0) {
    q.submit([&](sycl::handler &h) {
      auto src_acc = src_buf.template get_access<sycl::access::mode::read>(h);
      auto dest_acc =
          dest_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for<host_buffer_copy_kernel<T>>(
          src_buf.get_count(),
          [=](sycl::id<1> i) { dest_acc[i + offset] = src_acc[i]; });
    });
  }
}
template <typename T>
sycl::buffer<T, 1> buffer_resize(sycl::queue& q, sycl::buffer<T, 1> &buf,
                                 size_t new_size) {
  sycl::buffer<T, 1> new_buf(new_size);
  q.submit([&](sycl::handler &h) {
    auto acc = buf.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(acc.size(), [=](sycl::id<1> i) { new_acc[i] = acc[i]; });
  });
  return new_buf;
}

template <typename T, std::unsigned_integer uI_t = uint32_t>
sycl::buffer<T, 1>
device_buffer_combine(sycl::queue &q, sycl::buffer<T, 1> buf0,
                      sycl::buffer<T, 1> buf1, uI_t size0 = 0, uI_t size1 = 0) {
  if (size0 == 0) {
    size0 = buf0.get_count();
  }
  if (size1 == 0) {
    size1 = buf1.get_count();
  }
  sycl::buffer<T, 1> new_buf(size0 + size1);
  q.submit([&](sycl::handler &h) {
    auto acc0 = buf0.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(size0, [=](sycl::id<1> i) { new_acc[i] = acc0[i]; });
  });
  q.submit([&](sycl::handler &h) {
    auto acc1 = buf1.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(size1, [=](sycl::id<1> i) { new_acc[i + size0] = acc1[i]; });
  });
  return new_buf;
}

template <typename T, std::unsigned_integer uI_t = uint32_t>
inline void host_buffer_add(std::vector<sycl::buffer<T, 1> &> &bufs,
                            const std::vector<const std::vector<T> &> &vecs,
                            sycl::queue &q, const std::vector<uI_t> &offsets) {
  for (uI_t i = 0; i < vecs.size(); ++i) {
    host_buffer_add(bufs[i], vecs[i], q, offsets[i]);
  }
}
} // namespace Sycl_Graph
#endif