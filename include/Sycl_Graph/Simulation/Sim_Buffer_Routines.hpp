#ifndef SIM_BUFFER_ROUTINES_HPP
#define SIM_BUFFER_ROUTINES_HPP
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <CL/sycl.hpp>
// void single_community_state_accumulate(sycl::h_item<1> &it, const sycl::accessor<uint32_t, 1, sycl::access_mode::read> &vcm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc);

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, uint32_t Nt, sycl::range<1> compute_range, sycl::range<1> wg_range);

// std::vector<std::vector<std::vector<State_t>>> vector_remap(std::vector<State_t> &input, size_t N0, size_t N1, size_t N2);

// std::vector<std::vector<std::vector<uint32_t>>> vector_remap(std::vector<uint32_t> &input, size_t N0, size_t N1, size_t N2);

// std::vector<sycl::event> read_reset_buffers(
//     sycl::queue &q,
//     sycl::buffer<SIR_State, 3> &vertex_state,
//     sycl::buffer<uint32_t, 3> &events_from,
//     sycl::buffer<uint32_t, 3> &events_to,
//     sycl::buffer<uint32_t> &vcm,
//     uint32_t t,
//     std::vector<sycl::event> &dep_events,
//     sycl::range<1> compute_range,
//     sycl::range<1> wg_range);
void print_timestep(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<uint32_t, 3> &e_from, sycl::buffer<uint32_t, 3> &e_to, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, const Sim_Param &p);

#endif
