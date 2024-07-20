#pragma once
#include <SIR_SBM/utils/csv.hpp>
#include <sycl/sycl.hpp>
namespace SIR_SBM {

template <typename T, int N>
void validate_elements(sycl::queue &q, sycl::buffer<T, N> &buf, 
                        auto f,
                       auto to_str,
                       const char *msg = "",
                       sycl::event dep_event = {}) {
  auto vec = read_buffer<T, N>(q, buf, dep_event);
  auto pair_to_str = [](std::pair<uint32_t, uint32_t> p) {
    return std::to_string(p.first) + "," + std::to_string(p.second);
  };
  auto it = std::find_if_not(vec.begin(), vec.end(), f);
  if (it != vec.end()) {
    throw std::runtime_error(std::string("Validation failed at element ") +
                             std::to_string(std::distance(vec.begin(), it)) +
                             std::string(",value: ") + to_str(*it) +
                             std::string(", ") + msg);
  }
}
} // namespace SIR_SBM
