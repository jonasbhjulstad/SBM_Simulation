#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>
// inline void single_community_state_accumulate(sycl::nd_item<1> &it, const sycl::accessor<uint32_t, 2, sycl::access_mode::read> &vpm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc, uint32_t N_sims)
void single_community_state_accumulate(sycl::nd_item<1> &it, const auto &vpm_acc, const auto &v_acc, const auto &state_acc, auto N_sims, auto N_communities, auto N_vertices)
{
    auto Nt = v_acc.get_range()[1];
    auto sim_id = it.get_global_id()[0];
    for (int t = 0; t < Nt; t++)
    {
        uint32_t c_idx = 0;
        for (c_idx = 0; c_idx < N_communities; c_idx++)
        {
            state_acc[sim_id][t][c_idx] = {0, 0, 0};
        }
        for (int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            c_idx = vpm_acc[v_idx];
            auto v_state = v_acc[sim_id][t][v_idx];
            state_acc[sim_id][t][c_idx][v_state]++;
        }
    }
}
sycl::event accumulate_community_state(sycl::queue &q, sycl::event &dep_event, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t> &vpm_buf, sycl::buffer<State_t, 3> &community_buf, const sycl::nd_range<1>&nd_range)
{
    return q.submit([&](sycl::handler &h)
                    {
                h.depends_on(dep_event);
        auto v_acc = v_buf.template get_access<sycl::access::mode::read>(h);
        sycl::accessor<State_t, 3, sycl::access_mode::read_write> state_acc(community_buf, h);
        auto vpm_acc = vpm_buf.template get_access<sycl::access::mode::read>(h);
        auto N_sims = community_buf.get_range()[0];
        auto N_vertices = v_buf.get_range()[2];
        auto N_communities = community_buf.get_range()[2];
            h.parallel_for(nd_range, [=](sycl::nd_item<1> it)
            {
                single_community_state_accumulate(it, vpm_acc, v_acc, state_acc, N_sims, N_communities, N_vertices);
            }); });
}

sycl::event shift_buffer(sycl::queue &q, sycl::buffer<SIR_State, 3> &buf)
{
    auto N_sims = buf.get_range()[0];
    auto Nt = buf.get_range()[1];
    auto N_vertices = buf.get_range()[2];
    return q.submit([&](sycl::handler &h)
                    {
            auto start_acc = sycl::accessor<SIR_State, 3, sycl::access_mode::write>(buf, h, sycl::range<3>(N_sims,1, N_vertices), sycl::range<3>(0,0,0));
            auto end_acc = sycl::accessor<SIR_State, 3, sycl::access_mode::read>(buf, h, sycl::range<3>(N_sims,1, N_vertices), sycl::range<3>(0,Nt-1,0));
            h.copy(end_acc, start_acc); });
}

bool is_allocated_space_full(uint32_t t, uint32_t Nt_alloc)
{
    return ((t != 0) && (t % (Nt_alloc) == 0));
}
