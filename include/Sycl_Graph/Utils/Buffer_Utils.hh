#ifndef BUFFER_UTILS_HPP
#define BUFFER_UTILS_HPP
#include <Sycl_Graph/Utils/Common.hpp>
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
std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t N_rngs, uint32_t seed);

// std::vector<std::vector<float>> generate_floats(uint32_t rows, uint32_t cols, float min, float max, uint32_t seed);
// std::vector<std::vector<std::vector<float>>> generate_floats(uint32_t N0, uint32_t N1, uint32_t N2, float min, float max, uint32_t seed);

std::tuple<sycl::range<1>, sycl::range<1>> sim_ranges(sycl::queue& q, uint32_t N_sims);
// sycl::event move_buffer_row(sycl::queue& q, sycl::buffer<SIR_State,3>& buf, uint32_t row, std::vector<sycl::event>& dep_events);


extern template sycl::event initialize_device_buffer<uint32_t, 1>(sycl::queue& q, const std::vector<uint32_t> &host_data, sycl::buffer<uint32_t, 1>& buf);
extern template sycl::event initialize_device_buffer<uint32_t, 2>(sycl::queue& q, const std::vector<uint32_t> &host_data, sycl::buffer<uint32_t, 2>& buf);
extern template sycl::event initialize_device_buffer<uint32_t, 3>(sycl::queue& q, const std::vector<uint32_t> &host_data, sycl::buffer<uint32_t, 3>& buf);

extern template sycl::event initialize_device_buffer<float, 1>(sycl::queue& q, const std::vector<float> &host_data, sycl::buffer<float, 1>& buf);
extern template sycl::event initialize_device_buffer<float, 2>(sycl::queue& q, const std::vector<float> &host_data, sycl::buffer<float, 2>& buf);
extern template sycl::event initialize_device_buffer<float, 3>(sycl::queue& q, const std::vector<float> &host_data, sycl::buffer<float, 3>& buf);

extern template sycl::buffer<SIR_State, 3> create_device_buffer<SIR_State, 3>(sycl::queue& q, const std::vector<SIR_State> &host_data, const sycl::range<3>& range, sycl::event& event);
extern template sycl::event read_buffer<SIR_State, 3>(sycl::buffer<SIR_State,3>& buf, sycl::queue& q, std::vector<SIR_State>& result, std::vector<sycl::event>& dep_events);
extern template sycl::event read_buffer<State_t, 3>(sycl::buffer<State_t,3>& buf, sycl::queue& q, std::vector<State_t>& result, std::vector<sycl::event>& dep_events);

extern template sycl::event read_buffer<SIR_State, 3>(sycl::buffer<SIR_State,3>& buf, sycl::queue& q, std::vector<SIR_State>& result, sycl::event& dep_event);
extern template sycl::event read_buffer<State_t, 3>(sycl::buffer<State_t,3>& buf, sycl::queue& q, std::vector<State_t>& result, sycl::event& dep_event);

extern template sycl::event read_buffer<SIR_State, 3>(sycl::buffer<SIR_State,3>& buf, sycl::queue& q, std::vector<SIR_State>& p_result, std::vector<sycl::event>& dep_events, sycl::range<3> range, sycl::range<3> offset);
extern template sycl::event read_buffer<State_t, 3>(sycl::buffer<State_t,3>& buf, sycl::queue& q, std::vector<State_t>& p_result, std::vector<sycl::event>& dep_events, sycl::range<3> range, sycl::range<3> offset);

extern template sycl::event clear_buffer<uint32_t, 3>(sycl::queue& q, sycl::buffer<uint32_t, 3>& buf, std::vector<sycl::event>& dep_events);




#endif
