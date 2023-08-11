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
sycl::buffer<Static_RNG::default_rng> generate_rngs(sycl::queue& q, uint32_t N_rng, uint32_t seed, sycl::event& event);
sycl::buffer<Static_RNG::default_rng, 2> generate_rngs(sycl::queue& q, sycl::range<2> size, uint32_t seed, sycl::event& event);

std::vector<uint32_t> pairlist_to_vec(const std::vector<std::pair<uint32_t, uint32_t>> &pairlist);
std::vector<std::pair<uint32_t, uint32_t>> vec_to_pairlist(const std::vector<uint32_t> &vec);
sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed, sycl::event&);
std::vector<uint32_t> generate_seeds(uint32_t N_rng, uint32_t seed);


extern template std::vector<std::vector<uint32_t>> diff<uint32_t>(const std::vector<std::vector<uint32_t>> &v);
extern template std::vector<std::vector<int>> diff<int>(const std::vector<std::vector<int>> &v);

std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t seed);
std::vector<std::vector<float>> generate_floats(uint32_t rows, uint32_t cols, float min, float max, uint32_t seed);
std::vector<std::vector<std::vector<float>>> generate_floats(uint32_t N0, uint32_t N1, uint32_t N2, float min, float max, uint32_t seed);

extern template sycl::buffer<uint32_t, 1> create_device_buffer<uint32_t, 1>(sycl::queue& q, const std::vector<uint32_t> &host_data, const sycl::range<1>& range, sycl::event& event);
extern template sycl::buffer<uint32_t, 2> create_device_buffer<uint32_t, 2>(sycl::queue& q, const std::vector<uint32_t> &host_data, const sycl::range<2>& range, sycl::event& event);
extern template sycl::buffer<uint32_t, 3> create_device_buffer<uint32_t, 3>(sycl::queue& q, const std::vector<uint32_t> &host_data, const sycl::range<3>& range, sycl::event& event);

extern template sycl::buffer<float, 1> create_device_buffer<float, 1>(sycl::queue& q, const std::vector<float> &host_data, const sycl::range<1>& range, sycl::event& event);
extern template sycl::buffer<float, 2> create_device_buffer<float, 2>(sycl::queue& q, const std::vector<float> &host_data, const sycl::range<2>& range, sycl::event& event);
extern template sycl::buffer<float, 3> create_device_buffer<float, 3>(sycl::queue& q, const std::vector<float> &host_data, const sycl::range<3>& range, sycl::event& event);

extern template sycl::buffer<SIR_State, 3> create_device_buffer<SIR_State, 3>(sycl::queue& q, const std::vector<SIR_State> &host_data, const sycl::range<3>& range, sycl::event& event);
extern template std::vector<SIR_State> read_buffer<SIR_State, 3>(sycl::buffer<SIR_State,3>& buf, sycl::queue& q, sycl::event& event);
extern template std::vector<State_t> read_buffer<State_t, 3>(sycl::buffer<State_t,3>& buf, sycl::queue& q, sycl::event& event);


#endif
