#ifndef SIM_BUFFER_ROUTINES_HPP
#define SIM_BUFFER_ROUTINES_HPP
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <CL/sycl.hpp>

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims);

#endif
