#ifndef BUFFER_UTILS_HPP
#define BUFFER_UTILS_HPP
#include <vector>
#include <cstdint>
#include <CL/sycl.hpp>
#include "Buffer_Utils_impl.hpp"
void linewrite(std::ofstream &file, const std::vector<uint32_t> &state_iter);

void linewrite(std::ofstream &file, const std::vector<float> &val);

void linewrite(std::ofstream &file,
               const std::vector<std::array<uint32_t, 3>> &state_iter);

std::vector<uint32_t> pairlist_to_vec(const std::vector<std::pair<uint32_t, uint32_t>> &pairlist);
std::vector<std::pair<uint32_t, uint32_t>> vec_to_pairlist(const std::vector<uint32_t> &vec);
sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed);
extern template std::vector<uint32_t> merge_vectors<uint32_t>(const std::vector<std::vector<uint32_t>> &);
extern template std::vector<float> merge_vectors<float>(const std::vector<std::vector<float>> &);
extern template sycl::buffer<uint32_t> buffer_create_1D<uint32_t>(sycl::queue &, const std::vector<uint32_t> &, sycl::event &);
extern template sycl::buffer<float> buffer_create_1D<float>(sycl::queue &, const std::vector<float> &, sycl::event &);
extern template sycl::buffer<uint32_t, 2> buffer_create_2D<uint32_t>(sycl::queue &, const std::vector<std::vector<uint32_t>> &, sycl::event &);
extern template sycl::buffer<float, 2> buffer_create_2D<float>(sycl::queue &, const std::vector<std::vector<float>> &, sycl::event &);
extern template sycl::buffer<SIR_State, 2> buffer_create_2D<SIR_State>(sycl::queue &, const std::vector<std::vector<SIR_State>> &, sycl::event &);

extern template std::vector<std::vector<uint32_t>> read_buffer<uint32_t>(sycl::queue &q, sycl::buffer<uint32_t, 2> &buf,
                                                                         sycl::event events);
extern template std::vector<std::vector<float>> read_buffer<float>(sycl::queue &q, sycl::buffer<float, 2> &buf, sycl::event events);

extern template std::vector<std::vector<uint32_t>> diff<uint32_t>(const std::vector<std::vector<uint32_t>> &v);
extern template std::vector<std::vector<int>> diff<int>(const std::vector<std::vector<int>> &v);

#endif
