#ifndef COMMUNITY_STATE_HPP
#define COMMUNITY_STATE_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>

// SYCL_EXTERNAL void single_community_state_accumulate(sycl::h_item<1> &it, const auto &vcm_acc, const auto &v_acc, auto &state_acc);

sycl::event accumulate_community_state(sycl::queue& q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3>& v_buf, sycl::buffer<uint32_t>& vcm_buf, sycl::buffer<State_t, 3> community_buf, uint32_t Nt, sycl::range<1> compute_range, sycl::range<1> wg_range);
void print_community_state(sycl::queue& q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3>& v_buf, sycl::buffer<uint32_t>& vcm_buf, uint32_t Nt, uint32_t N_communities, sycl::range<1> compute_range, sycl::range<1> wg_range);

#endif
