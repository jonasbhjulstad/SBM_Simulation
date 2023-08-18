#include <Sycl_Graph/Buffer_Validation.hpp>
#include <Sycl_Graph/Community_State.hpp>
#include <iostream>
void single_community_state_accumulate(sycl::h_item<1> &it, const sycl::accessor<uint32_t, 1, sycl::access_mode::read> &vcm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc)
{
    auto Nt = v_acc.get_range()[0];
    auto N_vertices = vcm_acc.get_range()[0];
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

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t> &vcm_buf, sycl::buffer<State_t, 3> community_buf, uint32_t Nt, sycl::range<1> compute_range, sycl::range<1> wg_range)
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
            gr.parallel_for_work_item([&](sycl::h_item<1> it)
            {
                single_community_state_accumulate(it, vcm_acc, v_acc, state_acc);
            });
        }); });
}

std::vector<std::vector<std::vector<State_t>>> vector_remap(std::vector<State_t> &input, size_t N0, size_t N1, size_t N2)
{
    size_t M = N0 * N1 * N2;
    if (input.size() != M)
    {
        throw std::runtime_error("Input vector size does not match 3D dimensions.");
    }

    std::vector<std::vector<std::vector<State_t>>> output(N0, std::vector<std::vector<State_t>>(N1, std::vector<State_t>(N2)));

    // linear position defined by i2 + i1*n2 + i0*n1*n2
    for (size_t i0 = 0; i0 < N0; i0++)
    {
        for (size_t i1 = 0; i1 < N1; i1++)
        {
            for (size_t i2 = 0; i2 < N2; i2++)
            {
                output[i0][i1][i2] = input[i2 + i1 * N2 + i0 * N1 * N2];
            }
        }
    }
    return output;
}

void print_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t> &vcm_buf, uint32_t Nt, uint32_t N_communities, sycl::range<1> compute_range, sycl::range<1> wg_range)
{
    auto N_sims = compute_range[0] * wg_range[0];
    sycl::buffer<State_t, 3> community_buf(sycl::range<3>(Nt, N_sims, N_communities));
    accumulate_community_state(q, dep_events, v_buf, vcm_buf, community_buf, Nt, compute_range, wg_range).wait();
    std::vector<State_t> community_state_flat(Nt * N_sims * N_communities);
    q.submit([&](sycl::handler &h)
             {
        h.depends_on(dep_events);
        auto community_acc = community_buf.template get_access<sycl::access::mode::read>(h);
        h.copy(community_acc, community_state_flat.data()); })
        .wait();
    auto community_state = vector_remap(community_state_flat, Nt, N_sims, N_communities);
    for (size_t i2 = 0; i2 < Nt; i2++)
    {
        std::cout << "t_alloc = " << i2 << "\t| ";
        for (size_t i0 = 0; i0 < std::min({(uint32_t)N_sims, (uint32_t)3}); i0++)
        {
            for (size_t i1 = 0; i1 < N_communities; i1++)
            {
                std::cout << community_state[i2][i0][i1][0] << " " << community_state[i2][i0][i1][1] << " " << community_state[i2][i0][i1][2] << " | ";
            }
        }
        std::cout << std::endl;
    }
}
