#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <sycl/sycl.hpp>
#end
namespace SIR_SBM {
template <typename T, int N = 1>
sycl::event buffer_copy(sycl::queue &q, sycl::buffer<T, N> &buf,
                        const std::vector<T> &data) {
  if (data.size() != buf.size()) {
    throw std::runtime_error("Data size is not equal to buffer size");
  }
  return q.submit([&](sycl::handler &cgh) {
    auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
    cgh.copy(data.data(), acc);
  });
}
#hdr
template <typename T, int N = 1>
void buffer_copy(sycl::queue &q, sycl::buffer<T, N> &buf,
                 const std::vector<T> &&data) {
  if (data.size() != buf.size()) {
    throw std::runtime_error("Data size is not equal to buffer size");
  }
  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
     cgh.copy(data.data(), acc);
   }).wait();
}

template <typename T, int N = 1>
sycl::event buffer_fill(sycl::queue& q, sycl::buffer<T, N>& buf, T val) {
  return q.submit([&](sycl::handler& h) {
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    h.fill(acc, val);
  });
}

template <typename T, int N = 1>
sycl::buffer<T, N> make_buffer(sycl::queue &q, const std::vector<T> &&data,
                               sycl::range<N> r) {
  sycl::buffer<T, N> buf(r);
  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
     cgh.copy(data.data(), acc);
   }).wait();
  return buf;
}
#end
template <int N> void validate_range(sycl::range<N> r, sycl::range<N> r_buf) {
  for (uint32_t i = 0; i < N; i++) {
    if (r[i] > r_buf[i]) {
      throw std::runtime_error("Ranges do not match at idx " +
                               std::to_string(i) + " " + std::to_string(r[i]) +
                               " " + std::to_string(r_buf[i]));
    }
  }
}

template <typename T>
std::tuple<uint32_t, uint32_t, uint32_t> get_range(const sycl::buffer<T, 3> &buf) {
  return std::make_tuple(buf.get_range()[0], buf.get_range()[1],
                         buf.get_range()[2]);
}

// GET RANGE FOR 2D BUFFER
template <typename T>
std::tuple<uint32_t, uint32_t> get_range(const sycl::buffer<T, 2> &buf) {
  return std::make_tuple(buf.get_range()[0], buf.get_range()[1]);
}

template <typename T>
std::tuple<uint32_t> get_range(const sycl::buffer<T, 1> &buf) {
  return std::make_tuple(buf.get_range()[0]);
}

template <typename T, uint32_t N>
sycl::event read_buffer(sycl::queue &q, sycl::buffer<T, N> &buf,
                        std::vector<T> &data, sycl::event dep_event) {

  data.resize(buf.size());
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto acc = buf.template get_access<sycl::access::mode::read>(h);
    h.copy(acc, data.data());
  });
}
// runs inplace infection on vertices at time t
sycl::event zero_fill(sycl::queue &q, sycl::buffer<uint32_t, 3> &buf,
                      sycl::range<3> range, sycl::range<3> offset,
                      sycl::event dep_event = {}) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto acc = sycl::accessor<uint32_t, 3, sycl::access::mode::read_write>(
        buf, h, range, offset);
    h.parallel_for(acc.get_range(), [=](sycl::id<3> idx) { acc[idx] = 0; });
  });
}
template <typename T, uint32_t N>
std::vector<T> read_buffer(sycl::queue &q, sycl::buffer<T, N> &buf,
                           sycl::event dep_event) {
  std::vector<T> data;
  read_buffer<T, N>(q, buf, data, dep_event).wait();
  return data;
}

template <typename T> sycl::buffer<T> dummy_buf_1() {
  return sycl::buffer<T>(sycl::range<1>(1));
}

template <typename T> sycl::buffer<T, 2> dummy_buf_2() {
  return sycl::buffer<T, 2>(sycl::range<2>(1, 1));
}

template <typename T> sycl::buffer<T, 3> dummy_buf_3() {
  return sycl::buffer<T, 3>(sycl::range<3>(1, 1, 1));
}

} // namespace SIR_SBM