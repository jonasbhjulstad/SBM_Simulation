#ifndef SBM_SIMULATION_SIMULATION_STATE_ACCUMULATION_HPP
#define SBM_SIMULATION_SIMULATION_STATE_ACCUMULATION_HPP
#include <CL/sycl.hpp>
#include <SBM_Database/Simulation/SIR_Types.hpp>
// SYCL_EXTERNAL void single_community_state_accumulate(sycl::nd_item<1> &it, const sycl::accessor<uint32_t, 2, sycl::access_mode::read> &vcm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc, uint32_t N_sims);

sycl::event accumulate_community_state(sycl::queue &q, sycl::event &dep_event, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t> &vcm_buf, sycl::buffer<State_t, 3> &community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims, uint32_t t_start = 0, uint32_t t_end = 0);
sycl::event move_buffer_row(sycl::queue &q, sycl::buffer<SIR_State, 3> &buf, uint32_t row, sycl::event &dep_events);

bool is_allocated_space_full(uint32_t t, uint32_t Nt_alloc);
// void read_vcm(sycl::queue& q, auto& buf);

#endif
