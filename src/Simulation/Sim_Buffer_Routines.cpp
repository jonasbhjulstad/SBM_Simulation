#include <iostream>
#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>

void single_community_state_accumulate(sycl::h_item<1> &it, const auto &vcm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc)
{
    auto Nt = v_acc.get_range()[0];
    auto N_vertices = v_acc.get_range()[2];
    auto sim_id = it.get_global_id(0);
    auto lid = it.get_local_id(0);
    auto N_communities = state_acc.get_range()[2];
    for (int t = 0; t < Nt; t++)
    {
        for (int c_idx = 0; c_idx < N_communities; c_idx++)
        {
            state_acc[t][sim_id][c_idx] = {0, 0, 0};
        }
        for (int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto c_idx = vcm_acc[v_idx];
            auto v_state = v_acc[t][sim_id][v_idx];
            state_acc[t][sim_id][c_idx][v_state]++;
        }
    }
}

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, uint32_t Nt, sycl::range<1> compute_range, sycl::range<1> wg_range)
{
    auto range = sycl::range<3>(Nt, v_buf.get_range()[1], v_buf.get_range()[2]);
    return q.submit([&](sycl::handler &h)
                    {
                h.depends_on(dep_events);
        auto v_acc = construct_validate_accessor<SIR_State, 3, sycl::access_mode::read>(v_buf, h, range);
        sycl::accessor<State_t, 3, sycl::access_mode::read_write> state_acc(community_buf, h);
        auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read>(h);
        h.parallel_for_work_group(compute_range, wg_range, [state_acc, vcm_acc, v_acc](sycl::group<1> gr)
        {
            auto graph_id = gr.get_group_id();
            gr.parallel_for_work_item([&](sycl::h_item<1> it)
            {
                single_community_state_accumulate(it, vcm_acc[graph_id], v_acc, state_acc);
            });
        }); });
}



void print_timestep(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<uint32_t, 3> &e_from, sycl::buffer<uint32_t, 3> &e_to, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, const Sim_Param &p)
{
    auto N_sims = p.compute_range[0] * p.wg_range[0];
    auto N_connections = e_from.get_range()[2];
    auto Nt_alloc = v_buf.get_range()[0]-1;
    auto N_communities = p.N_communities;
    sycl::buffer<State_t, 3> community_buf(sycl::range<3>(Nt_alloc + 1, N_sims, N_communities));

    accumulate_community_state(q, dep_events, v_buf, vcm_buf, community_buf, Nt_alloc + 1, p.compute_range, p.wg_range).wait();
    std::vector<State_t> community_state_flat((Nt_alloc+1) * N_sims * N_communities);
    q.submit([&](sycl::handler &h)
             {
        h.depends_on(dep_events);
        auto community_acc = community_buf.template get_access<sycl::access::mode::read>(h);
        h.copy(community_acc, community_state_flat.data()); })
        .wait();
    auto community_state = vector_remap(community_state_flat, Nt_alloc+1, N_sims, N_communities);

    auto read_e_buf = [&](auto &buf)
    {
        std::vector<uint32_t> res_flat(Nt_alloc * N_sims * N_connections);
        q.submit([&](sycl::handler &h)
                 {
        h.depends_on(dep_events);
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, res_flat.data()); })
            .wait();
        return vector_remap(res_flat, Nt_alloc, N_sims, N_connections);
    };
    auto from_vec = read_e_buf(e_from);
    auto to_vec = read_e_buf(e_to);

    for (size_t i2 = 0; i2 < Nt_alloc; i2++)
    {
        std::cout << "t_alloc = " << i2 << "\t| ";
        for (size_t i0 = 0; i0 < std::min({(uint32_t)N_sims, (uint32_t)3}); i0++)
        {
            for (size_t i1 = 0; i1 < N_communities; i1++)
            {
                std::cout << community_state[i2][i0][i1][0] << " " << community_state[i2][i0][i1][1] << " " << community_state[i2][i0][i1][2] << " | ";
            }
            std::cout << "\t\t";
            for (size_t i1 = 0; i1 < N_connections; i1++)
            {
                std::cout << from_vec[i2][i0][i1] << " " << to_vec[i2][i0][i1] << " | ";
            }
        }
        std::cout << std::endl;
    }
}
