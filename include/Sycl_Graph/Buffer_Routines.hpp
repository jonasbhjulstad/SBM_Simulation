#ifndef SYCL_GRAPH_BUFFER_ROUTINES_HPP
#define SYCL_GRAPH_BUFFER_ROUTINES_HPP

#include <CL/sycl.hpp>
#include <vector>

template <typename T> void print_buffer(sycl::buffer<T, 1> &buf) {
  auto acc = buf.get_host_access();
  for (int i = 0; i < buf.size(); i++) {
    std::cout << acc[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
sycl::event copy_to_buffer(sycl::buffer<T, 1> &buf, const std::vector<T> &vec,
                           sycl::queue &q) {
  assert(buf.size() == vec.size());
  sycl::buffer<T, 1> tmp(vec.data(), sycl::range<1>(vec.size()));
  return q.submit([&](sycl::handler &h) {
    auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(acc.size(), [=](sycl::id<1> id) { acc[id] = tmp_acc[id]; });
    // h.copy(tmp_acc, acc);
  });
}
template <typename T>
auto initialize_buffer_vector(const std::vector<std::vector<T>> &init_state,
                              sycl::queue &q) {
  uint32_t vec_size = init_state.size();
  uint32_t buf_size = init_state[0].size();
  auto buf_vec = std::vector<sycl::buffer<T>>(
      vec_size, sycl::buffer<T>(sycl::range<1>(buf_size)));

  std::vector<sycl::event> events(vec_size);

  std::transform(buf_vec.begin(), buf_vec.end(), init_state.begin(),
                 events.begin(), [&](auto &buf, const auto &state) { return copy_to_buffer(buf, state, q);});
  return std::make_tuple(buf_vec, events);
}

template <typename T>
auto initialize_buffer_vector(uint32_t buf_size, uint32_t vec_size,
                              const T init_state, sycl::queue &q) {
  auto init_state_vec = std::vector<std::vector<T>>(
      vec_size, std::vector<T>(buf_size, init_state));
  return initialize_buffer_vector(buf_size, vec_size, init_state_vec, q);
}

template <typename T>
sycl::event copy_to_buffer(sycl::buffer<T, 1> &buf, const std::vector<T> &&vec,
                           sycl::queue &q) {
  return copy_to_buffer(buf, vec, q);
}

#endif