#ifndef SBM_SIMULATION_SIMULATION_STATE_ACCUMULATION_HPP
#define SBM_SIMULATION_SIMULATION_STATE_ACCUMULATION_HPP
#include <CL/sycl.hpp>
#include <SBM_Database/Simulation/SIR_Types.hpp>
sycl::event accumulate_community_state(sycl::queue &q, sycl::event &dep_event, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t> &vpm_buf, sycl::buffer<State_t, 3> &community_buf, const sycl::nd_range<1>&nd_range);

sycl::event shift_buffer(sycl::queue &q, sycl::buffer<SIR_State, 3> &buf);

bool is_allocated_space_full(uint32_t t, uint32_t Nt_alloc);
// void read_vpm(sycl::queue& q, auto& buf);

#endif
