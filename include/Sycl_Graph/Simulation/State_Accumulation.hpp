#ifndef SYCL_GRAPH_SIMULATION_STATE_ACCUMULATION_HPP
#define SYCL_GRAPH_SIMULATION_STATE_ACCUMULATION_HPP
void single_community_state_accumulate(sycl::nd_item<1> &it, const auto &vcm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc, uint32_t N_sims);

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims);
sycl::event move_buffer_row(sycl::queue &q, sycl::buffer<SIR_State, 3> &buf, uint32_t row, std::vector<sycl::event> &dep_events);

bool is_allocated_space_full(uint32_t t, uint32_t Nt_alloc);

#endif
