// sycl_validate.hpp
//

#ifndef LZZ_sycl_validate_hpp
#define LZZ_sycl_validate_hpp
#include <SIR_SBM/sycl_routines.hpp>
namespace SIR_SBM{
template <typename T>
using string_f = std::string (*)(T);
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
} // namespace SIR_SBM
#define LZZ_INLINE inline
#undef LZZ_INLINE
#endif
