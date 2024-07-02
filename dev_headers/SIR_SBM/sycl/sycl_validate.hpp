#pragma once
#hdr
#include <SIR_SBM/utils/csv.hpp>
#include <SIR_SBM/sycl/sycl_routines.hpp>
namespace SIR_SBM {
template <typename T> using string_f = std::string (*)(T);
template <typename T, int N, string_f<T> str_f = std::to_string>
void validate_elements(sycl::queue &q, sycl::buffer<T, N> &buf, auto f,
                       const char *msg = "", sycl::event dep_event = {}) {
  auto vec = read_buffer<T, N>(q, buf, dep_event);
  auto it = std::find_if_not(vec.begin(), vec.end(), f);
  if (it != vec.end()) {
    throw std::runtime_error(std::string("Validation failed at element ") +
                             std::to_string(std::distance(vec.begin(), it)) +
                             std::string(",value: ") + str_f(*it) +
                             std::string(", ") + msg);
  }
}
template <typename T>
void buffer_log(sycl::queue &q, sycl::buffer<T, 2> &buf, const char *fname,
                sycl::event dep_event = {}) {
#ifdef SIR_SBM_BUFFER_LOG
  auto vec = read_buffer<T, 2>(q, buf, dep_event);
  write_csv(vec, fname, buf.get_range()[0], buf.get_range()[1]);
#endif
}
template <typename T>
void buffer_log(sycl::queue &q, sycl::buffer<T, 3> &buf,
                const char *fname_prefix, sycl::event dep_event = {}) {
#ifdef SIR_SBM_BUFFER_LOG
  auto vec = read_buffer<T, 3>(q, buf, dep_event);
  for (uint32_t i = 0; i < buf.get_range()[0]; i++) {
    write_csv(vec[i],
              std::string(fname_prefix) + std::string("_") + std::to_string(i) +
                  std::string(".csv"),
              buf.get_range()[1], buf.get_range()[2]);
  }
#endif
}
} // namespace SIR_SBM
#end
