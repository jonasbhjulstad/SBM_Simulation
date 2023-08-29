#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>
#include <iostream>


void single_community_state_accumulate(sycl::nd_item<1> &it, const auto &vcm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc)
{
    auto Nt = v_acc.get_range()[0];
    auto N_vertices = v_acc.get_range()[2];
    auto N_sims = v_acc.get_range()[1];
    auto sim_id = it.get_global_id()[0];
    auto graph_id = static_cast<uint32_t>(std::floor(static_cast<double>(sim_id) / static_cast<double>(N_sims)));
    auto N_communities = state_acc.get_range()[2];
    for (int t = 0; t < Nt; t++)
    {
        for (int c_idx = 0; c_idx < N_communities; c_idx++)
        {
            state_acc[t][sim_id][c_idx] = {0, 0, 0};
        }
        for (int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto c_idx = vcm_acc[graph_id][v_idx];
            auto v_state = v_acc[t][sim_id][v_idx];
            state_acc[t][sim_id][c_idx][v_state]++;
        }
    }
}

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range)
{
    auto Nt = v_buf.get_range()[0];
    auto range = sycl::range<3>(Nt, v_buf.get_range()[1], v_buf.get_range()[2]);
    return q.submit([&](sycl::handler &h)
                    {
                h.depends_on(dep_events);
        auto v_acc = construct_validate_accessor<SIR_State, 3, sycl::access_mode::read>(v_buf, h, range);
        sycl::accessor<State_t, 3, sycl::access_mode::read_write> state_acc(community_buf, h);
        auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read>(h);

            h.parallel_for(sycl::nd_range<1>(compute_range, wg_range), [=](sycl::nd_item<1> it)
            {

                single_community_state_accumulate(it, vcm_acc, v_acc, state_acc);
            }); });
}
