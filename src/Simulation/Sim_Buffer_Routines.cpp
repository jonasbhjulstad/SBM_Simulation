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
    auto v_dt = p.N_sims*N_vertices;
    auto v_d_sim = it[0]*N_vertices;

    auto c_dt = p.N_sims*N_communities;
    auto c_d_sim = it[0]*N_communities;


    for (int t = 0; t < Nt; t++)
    {
        for (int c_idx = 0; c_idx < N_communities; c_idx++)
        {
            state_acc[c_dt*t + c_d_sim + c_idx] = {0, 0, 0};
        }
        for (int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto c_idx = vcm[v_idx];
            auto v_state = v_acc[v_dt*t + v_d_sim + v_idx];
            state_acc[c_dt*t + c_d_sim + c_idx]++;
        }
    }
}

sycl::event accumulate_community_state(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
{
    return q.submit([&](sycl::handler &h)
                    {
        h.depends_on(dep_events);
        h.parallel_for(p.N_sims, [=](sycl::item<1> it)
                       { single_community_state_accumulate(it, b.vcm, b.vertex_state, b.community_state, p.N_communities, p.N_vertices(), p.Nt); });
}



void print_timestep(sycl::queue &q, const Sim_Param& p, Sim_Buffers& b, std::vector<sycl::event> &dep_events )
{
        auto N_sims = p.compute_range[0] * p.wg_range[0];
        auto N_connections = e_from.get_range()[2];
        auto Nt_alloc = v_buf.get_range()[0] - 1;
        auto N_communities = p.N_communities;

        accumulate_community_state(q, p, b, dep_events);
        // accumulate_community_state(q, dep_events, v_buf, vcm_buf, community_buf, Nt_alloc + 1, p.compute_range, p.wg_range).wait();
        std::vector<State_t> community_state_flat((Nt_alloc + 1) * N_sims * N_communities);
        q.submit([&](sycl::handler &h)
                 {
        h.depends_on(dep_events);
        auto community_acc = community_buf.template get_access<sycl::access::mode::read>(h);
        h.copy(community_acc, community_state_flat.data()); })
            .wait();
        auto community_state = vector_remap(community_state_flat, Nt_alloc + 1, N_sims, N_communities);

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
