
#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>

SYCL_EXTERNAL auto floor_div(auto a, auto b) { return static_cast<uint32_t>(std::floor(static_cast<float>(a) / static_cast<float>(b))); }

std::vector<sycl::event> recover(sycl::queue &q,
                                 const Sim_Param &p,
                                 sycl::buffer<SIR_State, 3> &vertex_state,
                                 sycl::buffer<Static_RNG::default_rng> &rngs,
                                 uint32_t t,
                                 std::vector<sycl::event> &dep_event)
{
    float p_R = p.p_R;
    uint32_t N_vertices = vertex_state.get_range()[2];
    uint32_t N_sims = p.N_sims;
    auto Nt_alloc = vertex_state.get_range()[0] - 1;
    auto t_alloc = t % Nt_alloc;
    std::size_t global_mem_acc_size = rngs.byte_size() + vertex_state.byte_size();
    std::size_t local_mem_acc_size = 0; // p.wg_range[0] * sizeof(Static_RNG::default_rng);
    assert(global_mem_acc_size < p.global_mem_size);
    // assert(local_mem_acc_size < p.local_mem_size);
    auto cpy_event = q.submit([&](sycl::handler &h)
                              {
    h.depends_on(dep_event);
    auto rng_acc_glob = rngs.template get_access<sycl::access_mode::read_write>(h);
    auto v_glob_prev = construct_validate_accessor<SIR_State, 3, sycl::access_mode::read>(vertex_state, h, sycl::range<3>(1, p.N_graphs*N_sims, N_vertices), sycl::range<3>(t_alloc, 0, 0));
    auto v_glob_next = construct_validate_accessor<SIR_State, 3, sycl::access_mode::write>(vertex_state, h, sycl::range<3>(1, p.N_graphs*N_sims, N_vertices), sycl::range<3>(t_alloc + 1, 0, 0));
                        h.depends_on(dep_event);
                        h.copy(v_glob_prev, v_glob_next); });

    auto event = q.submit([&](sycl::handler &h)
                          {
                              h.depends_on(cpy_event);
                              auto rng_acc_glob = rngs.template get_access<sycl::access_mode::read_write>(h);
                              auto v_glob = construct_validate_accessor<SIR_State, 3, sycl::access_mode::read_write>(vertex_state, h, sycl::range<3>(1, N_sims, N_vertices), sycl::range<3>(t_alloc + 1, 0, 0));
                              h.parallel_for(sycl::nd_range<1>(p.compute_range, p.wg_range), [=](sycl::nd_item<1> it)
                                             {
            auto sim_id = it.get_global_id();

        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
        for(int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto state_prev = v_glob[0][sim_id][v_idx];
            if (state_prev == SIR_INDIVIDUAL_I) {
                if (bernoulli_R(rng_acc_glob[sim_id])) {
                v_glob[0][sim_id][v_idx] = SIR_INDIVIDUAL_R;
                }
            }
            } });
                          });
    return {event};
}

sycl::event initialize_vertices(sycl::queue &q, const Sim_Param &p,
                                sycl::buffer<SIR_State, 3> &vertex_state,
                                sycl::buffer<Static_RNG::default_rng> &rngs)
{
    uint32_t N_vertices = vertex_state.get_range()[2];
    float p_I0 = p.p_I0;
    float p_R0 = p.p_R0;
    auto N_sims = p.N_sims;
    auto buf_size = vertex_state.get_range()[1] * vertex_state.get_range()[2] * sizeof(SIR_State) + rngs.byte_size();
    assert(buf_size < p.global_mem_size);
    auto init_event = q.submit([&](sycl::handler &h)
                               {
    auto v_acc = construct_validate_accessor<SIR_State, 3, sycl::access::mode::write>(vertex_state, h, sycl::range<3>(1, p.N_sims, N_vertices), sycl::range<3>(0, 0, 0));
    auto rng_acc =
        rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::nd_range<1>(p.compute_range, p.wg_range), [=](sycl::nd_item<1> it)
        {

            auto sim_id = it.get_global_id();
            for(int vertex_idx = 0; vertex_idx < N_vertices; vertex_idx++)
            {
            auto& rng = rng_acc[sim_id];
            Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
            Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);
            if (bernoulli_I(rng)) {
                v_acc[0][sim_id][vertex_idx] = SIR_INDIVIDUAL_I;
            } else if (bernoulli_R(rng)) {
                v_acc[0][sim_id][vertex_idx] = SIR_INDIVIDUAL_R;
            } else {
                v_acc[0][sim_id][vertex_idx] = SIR_INDIVIDUAL_S;
            }
            } }); });

    return init_event;
}


std::vector<sycl::event> infect(sycl::queue &q,
                                const Sim_Param &p,
                                Sim_Buffers &b,
                                uint32_t t,
                                std::vector<sycl::event> &dep_event)
{

    uint32_t N_edges = b.ecm.size();
    uint32_t N_vertices = b.vertex_state.get_range()[2];
    uint32_t N_sims = p.N_sims;
    uint32_t t_alloc = t % p.Nt_alloc;
    uint32_t Nt = p.Nt;
    std::size_t global_mem_acc_size = b.ecm.byte_size() + b.p_Is.byte_size() + b.rngs.byte_size() + b.vertex_state.byte_size() + b.events_to.byte_size() + b.events_from.byte_size();
    assert(global_mem_acc_size < p.global_mem_size);
    // assert(local_mem_acc_size < p.local_mem_size);
    auto inf_event = q.submit([&](sycl::handler &h)
                              {
                                  h.depends_on(dep_event);
                                  auto ecm_acc = b.ecm.template get_access<sycl::access::mode::read>(h);
                                  auto p_I_acc_glob = construct_validate_accessor<float, 3, sycl::access::mode::read>(b.p_Is, h, sycl::range<3>(1, N_sims*p.N_graphs, b.N_connections_max), sycl::range<3>(t, 0, 0));
                                  auto rng_acc_glob = b.rngs.template get_access<sycl::access::mode::read_write>(h);
                                  auto v_glob_next = construct_validate_accessor<SIR_State, 3, sycl::access_mode::write>(b.vertex_state, h, sycl::range<3>(1, N_sims*p.N_graphs, N_vertices), sycl::range<3>(t_alloc + 1, 0, 0));
                                  auto e_count = b.edge_counts.template get_access<sycl::access::mode::read>(h);
                                  auto e_offset = b.edge_offsets.template get_access<sycl::access::mode::read>(h);
                                  auto e_acc_from = b.edge_from.template get_access<sycl::access::mode::read>(h);
                                  auto e_acc_to = b.edge_to.template get_access<sycl::access::mode::read>(h);
                                  auto N_connections_acc = b.N_connections.template get_access<sycl::access::mode::read>(h);
                                  auto event_to_acc_glob = construct_validate_accessor<uint32_t, 3, sycl::access::mode::write>(b.events_to, h, sycl::range<3>(1, N_sims*p.N_graphs, b.N_connections_max), sycl::range<3>(t_alloc, 0, 0));
                                  auto event_from_acc_glob = construct_validate_accessor<uint32_t, 3, sycl::access::mode::write>(b.events_from, h, sycl::range<3>(1, N_sims*p.N_graphs, b.N_connections_max), sycl::range<3>(t_alloc, 0, 0));
                                  sycl::stream out(1024, 256, h);
                                  h.parallel_for(sycl::nd_range<1>(p.compute_range, p.wg_range), [=](sycl::nd_item<1> it)
                                                 {
            auto sim_id = it.get_global_id();
            auto graph_id = floor_div(sim_id, N_sims);
            auto edge_start_idx = e_offset[graph_id];
            auto edge_end_idx = edge_start_idx + e_count[graph_id];
            auto lid = it.get_local_id(0);
            uint32_t N_inf = 0;
            for (uint32_t edge_idx = edge_start_idx; edge_idx < edge_end_idx; edge_idx++)
            {
                auto connection_id = ecm_acc[edge_idx];
                auto v_from_id = e_acc_from[edge_idx];
                auto v_to_id = e_acc_to[edge_idx];
                const auto v_prev_from = v_glob_next[0][sim_id][v_from_id];
                const auto v_prev_to = v_glob_next[0][sim_id][v_to_id];

                if ((v_prev_from == SIR_INDIVIDUAL_S) && (v_prev_to == SIR_INDIVIDUAL_I))
                {
                    float p_I = p_I_acc_glob[0][sim_id][connection_id];
                    Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                    auto &rng = rng_acc_glob[sim_id]; // was lid
                    bernoulli_I.p = p_I;
                    if (bernoulli_I(rng))
                    {
                        N_inf++;
                        if (sim_id == 0)
                        {
                            out << "Infection on edge " << edge_idx << " to " << v_to_id << " from " << v_from_id << ", Connection " <<  connection_id << "\n";
                        }
                        v_glob_next[0][sim_id][v_from_id] = SIR_INDIVIDUAL_I;
                        event_from_acc_glob[0][sim_id][connection_id]++;
                    }
                }
                else if ((v_prev_from == SIR_INDIVIDUAL_I) && (v_prev_to == SIR_INDIVIDUAL_S))
                {
                    float p_I = p_I_acc_glob[0][sim_id][connection_id];
                    Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                    auto &rng = rng_acc_glob[sim_id]; // was lid
                    bernoulli_I.p = p_I;
                    if (bernoulli_I(rng))
                    {
                        if (sim_id == 0)
                        {
                            out << "Infection on edge " << edge_idx << " from " << v_from_id << " to " << v_to_id << ", Connection " <<  connection_id << "\n";
                        }
                        N_inf++;
                        v_glob_next[0][sim_id][v_to_id] = SIR_INDIVIDUAL_I;
                        event_to_acc_glob[0][sim_id][connection_id]++;
                    }
                }
            } });
                              });

    std::cout << "Timestep " << t << " done\n";
    return {inf_event};
}
