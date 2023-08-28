#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>
#include <iostream>

void single_community_state_accumulate(sycl::item<1> &it, const auto &vcm,
                                       const auto &vertex_state,
                                       auto &state_acc,
                                       uint32_t N_communities,
                                       uint32_t N_vertices,
                                       uint32_t Nt)
{

    auto N_sims = it.get_range()[0];

    for (int t = 0; t < Nt; t++)
    {
        auto v_offset = get_linear_offset(N_sims, Nt, N_vertices, it[0], t, 0);
        auto c_offset = get_linear_offset(N_sims, Nt, N_communities, it[0], t, 0);
        for (int c_idx = 0; c_idx < N_communities; c_idx++)
        {
            state_acc[c_offset + c_idx] = {0, 0, 0};
        }
        for (int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto c_idx = vcm[v_idx];
            auto v_state = vertex_state[v_offset + v_idx];
            state_acc[v_offset + c_idx][v_state]++;
        }
    }
}

sycl::event accumulate_community_state(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
{
    return q.submit([&](sycl::handler &h)
                    {
        h.depends_on(dep_events);
        h.parallel_for(p.N_sims, [=](sycl::item<1> it)
                       { single_community_state_accumulate(it, b.vcm, b.vertex_state, b.community_state, p.N_communities, p.N_vertices(), p.Nt);
                       }); });
}
