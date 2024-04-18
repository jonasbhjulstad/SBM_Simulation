#pragma once
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/reshape.hpp>
#include <sycl/sycl.hpp>
namespace SIR_SBM {
template <typename T, int N = 1>
sycl::event buffer_copy(sycl::queue &q, sycl::buffer<T, N> &buf,
                        const std::vector<T> &data) {
  if (data.size() < buf.size()) {
    throw std::runtime_error("Data size is less than buffer size");
  }
  return q.submit([&](sycl::handler &cgh) {
    auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
    cgh.copy(data.data(), acc);
  });
}

template <typename T, int N = 1>
sycl::event buffer_copy(sycl::queue &q, sycl::buffer<T, N> &buf,
                        const std::vector<T> &&data) {
  if (data.size() < buf.size()) {
    throw std::runtime_error("Data size is less than buffer size");
  }
  return q.submit([&](sycl::handler &cgh) {
    auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
    cgh.copy(data.data(), acc);
  });
}

template <size_t N>
void validate_range(sycl::range<N> r, sycl::range<N> r_buf) {
  for (size_t i = 0; i < N; i++) {
    if (r[i] > r_buf[i]) {
      throw std::runtime_error("Ranges do not match at idx " +
                               std::to_string(i));
    }
  }
}

template <typename T>
std::tuple<size_t, size_t, size_t> get_range(const sycl::buffer<T, 3> &buf) {
  return std::make_tuple(buf.get_range()[0], buf.get_range()[1],
                         buf.get_range()[2]);
}

template <typename T, size_t N>
sycl::event read_buffer(sycl::queue &q, sycl::buffer<T, N> &buf,
                        std::vector<T> &data, sycl::event dep_event) {
  auto [N_vertices, N_sims, Nt_alloc] = get_range(buf);
  data.resize(N_vertices * N_sims * Nt_alloc);
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto acc = buf.template get_access<sycl::access::mode::read>(h);
    h.copy(acc, data.data());
  });
}

template <typename T, size_t N>
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