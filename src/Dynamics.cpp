#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>
#include <Sycl_Graph/Dynamics.hpp>

std::vector<sycl::event> recover(sycl::queue &q,
                                 const Sim_Param &p,
                                 sycl::buffer<SIR_State, 3> &vertex_state,
                                 sycl::buffer<Static_RNG::default_rng> &rngs,
                                 uint32_t t,
                                 std::vector<sycl::event> &dep_event)
{
    float p_R = p.p_R;
    uint32_t N_vertices = p.N_pop * p.N_communities;
    uint32_t N_sims = p.N_sims;
    auto [compute_range, wg_range] = sim_ranges(q, p.N_sims);
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto rng_acc_glob = rngs.template get_access<sycl::access_mode::read_write>(h);
    auto v_glob_prev = construct_validate_accessor<SIR_State, 3, sycl::access_mode::read>(vertex_state, h, sycl::range<3>(1, N_sims, N_vertices), sycl::range<3>(t_alloc, 0, 0));
    auto v_glob_next = construct_validate_accessor<SIR_State, 3, sycl::access_mode::write>(vertex_state, h, sycl::range<3>(1, N_sims, N_vertices), sycl::range<3>(t_alloc + 1, 0, 0));
    sycl::local_accessor<Static_RNG::default_rng> rng_acc(wg_range, h);
    sycl::local_accessor<SIR_State, 2> v_prev(sycl::range<2>(wg_range[0], N_vertices), h);
    sycl::local_accessor<SIR_State, 2> v_next(sycl::range<2>(wg_range[0], N_vertices), h);

    h.parallel_for_work_group(compute_range, wg_range, [=](sycl::group<1> gr)
    {
        gr.parallel_for_work_item([&](sycl::h_item<1> it)
        {
            auto lid = it.get_local_id();
            auto gid = it.get_global_id();
            rng_acc[lid] = rng_acc_glob[gid];
            for(int v_idx = 0; v_idx < N_vertices; v_idx++)
            {
                v_prev[lid][v_idx] = v_glob_prev[0][gid][v_idx];
                v_next[lid][v_idx] = v_prev[lid][v_idx];
            }
        });
        gr.parallel_for_work_item([&](sycl::h_item<1> it)
        {
            auto sim_id = it.get_global_id(0);
            auto lid = it.get_local_id(0);

        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
        for(int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto state_prev = v_prev[lid][v_idx];
            if (state_prev == SIR_INDIVIDUAL_I) {
                if (bernoulli_R(rng_acc_glob[sim_id])) {
                v_next[lid][v_idx] = SIR_INDIVIDUAL_R;
                }
            }
            }
        });
        gr.parallel_for_work_item([&](sycl::h_item<1> it)
        {
            auto lid = it.get_local_id(0);
            auto gid = it.get_global_id(0);
            rng_acc_glob[gid] = rng_acc[lid];
            for(int v_idx = 0; v_idx < N_vertices; v_idx++)
            {
                v_glob_next[0][gid][v_idx] = v_next[lid][v_idx];
            }

        });

     }); });
    return {event};
}

sycl::event initialize_vertices(sycl::queue &q, const Sim_Param &p,
sycl::buffer<SIR_State, 3> &vertex_state,
sycl::buffer<Static_RNG::default_rng> &rngs)
{
    uint32_t N_vertices = p.N_pop * p.N_communities;
    float p_I0 = p.p_I0;
    float p_R0 = p.p_R0;
    auto [compute_range, wg_range] = sim_ranges(q, p.N_sims);
    std::vector<sycl::event> init_events(3);
    init_events[0] = q.submit([&](sycl::handler &h)
                              {
    auto v_acc = vertex_state.template get_access<sycl::access_mode::write>(h);
    auto rng_acc =
        rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for_work_group(compute_range, wg_range, [=](sycl::group<1> gr) {
        gr.parallel_for_work_item([&](sycl::h_item<1> it)
        {
            auto sim_id = it.get_global_id(0);
            auto lid = it.get_local_id(0);
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
            }
    }); }); });

    return init_events[2];
}

std::vector<sycl::event> infect(sycl::queue &q,
                                           const Sim_Param &p,
                                           Sim_Buffers &b,
                                           uint32_t t,
                                           std::vector<sycl::event> &dep_event)
{

    auto [compute_range, wg_range] = sim_ranges();
    uint32_t N_connections = b.events_from.get_range()[2];
    uint32_t N_edges = b.ecm.size();
    uint32_t N_vertices = N_vertices;
    uint32_t N_sims = p.N_sims;
    uint32_t t_alloc = t % p.Nt_alloc;
    uint32_t Nt = b.vertex_state.get_range()[0]-1;
    assert(N_vertices == 200);
    assert(b.ecm.size() == N_edges);
    assert(b.vcm.size() == N_vertices);
    assert(b.p_Is.get_range()[0] == Nt);
    assert(b.p_Is.get_range()[1] == N_sims);
    assert(b.p_Is.get_range()[2] == N_connections);
    assert(b.rngs.size() == N_sims);
    assert(b.vertex_state.get_range()[1] == N_sims);
    assert(b.vertex_state.get_range()[2] == N_vertices);
    assert(b.events_to.get_range()[1] == N_sims);
    // assert(events_to.get_range()[1] == Nt);
    assert(b.events_to.get_range()[2] == N_connections);
    assert(b.events_from.get_range()[1] == N_sims);
    // assert(events_from.get_range()[1] == Nt);
    assert(b.events_from.get_range()[2] == N_connections);
    auto inf_event = q.submit([&](sycl::handler &h)
                              {
        h.depends_on(dep_event);
        auto ecm_acc = b.ecm.template get_access<sycl::access::mode::read>(h);
        auto p_I_acc_glob = construct_validate_accessor<float, 3, sycl::access::mode::read>(b.p_Is, h, sycl::range<3>(1, N_sims, N_connections), sycl::range<3>(t, 0, 0));
        auto rng_acc_glob = b.rngs.template get_access<sycl::access::mode::read_write>(h);
        auto v_glob_next = construct_validate_accessor<SIR_State, 3, sycl::access_mode::write>(b.vertex_state, h, sycl::range<3>(1, N_sims, N_vertices), sycl::range<3>(t_alloc + 1, 0, 0));
        auto e_acc_0 = b.edge_to.template get_access<sycl::access::mode::read>(h);
        auto e_acc_1 = b.edge_from.template get_access<sycl::access::mode::read>(h);
        auto event_to_acc_glob = construct_validate_accessor<uint32_t, 3, sycl::access::mode::write>(b.events_to, h, sycl::range<3>(1, N_sims, N_connections), sycl::range<3>(t_alloc, 0, 0));
        auto event_from_acc_glob = construct_validate_accessor<uint32_t, 3, sycl::access::mode::write>(b.events_from, h, sycl::range<3>(1, N_sims, N_connections), sycl::range<3>(t_alloc, 0, 0));
        sycl::local_accessor<Static_RNG::default_rng> rng_acc(wg_range, h);
        sycl::local_accessor<SIR_State, 2> v_prev(sycl::range<2>(wg_range[0], N_vertices), h);
        sycl::local_accessor<SIR_State, 2> v_next(sycl::range<2>(wg_range[0], N_vertices), h);
        sycl::local_accessor<float, 2> p_I_acc(sycl::range<2>(wg_range[0], N_connections), h);
        sycl::local_accessor<uint32_t, 2> event_to_acc(sycl::range<2>(wg_range[0], N_connections), h);
        sycl::local_accessor<uint32_t, 2> event_from_acc(sycl::range<2>(wg_range[0], N_connections), h);
        auto local_mem_size_used = rng_acc.byte_size() + v_prev.byte_size() + v_next.byte_size() + p_I_acc.byte_size() + event_to_acc.byte_size() + event_from_acc.byte_size();
        // auto lms = this->local_mem_size();
        assert(local_mem_size_used < lms);
        sycl::stream out(1024*10, 256, h);
        h.parallel_for_work_group(compute_range, wg_range, [=](sycl::group<1> gr)
                                  {
                                    // Copy to local accessor
                                    gr.parallel_for_work_item([&](sycl::h_item<1> it)
                                    {
                                        auto sim_id = it.get_global_id(0);
                                        auto lid = it.get_local_id(0);
                                        for (int v_idx = 0; v_idx < N_vertices; v_idx++)
                                        {
                                            v_prev[lid][v_idx] = v_glob_next[0][sim_id][v_idx];
                                            v_next[lid][v_idx] = v_glob_next[0][sim_id][v_idx];
                                        }
                                        for(int i = 0; i < N_connections; i++)
                                        {
                                            p_I_acc[lid][i] = p_I_acc_glob[0][sim_id][i];
                                            event_from_acc[lid][i] = 0;
                                            event_to_acc[lid][i] = 0;
                                        }
                                        rng_acc[lid] = rng_acc_glob[sim_id];
                                    });

                                    gr.parallel_for_work_item([&](sycl::h_item<1> it)
                                                                  {
                                        auto sim_id = it.get_global_id(0);
                                        auto lid = it.get_local_id(0);
                                        uint32_t N_inf = 0;
                                          for (uint32_t edge_idx = 0; edge_idx < N_edges; edge_idx++)
                                          {
                                            auto connection_id = ecm_acc[edge_idx];
                                            auto v_from_id = e_acc_0[edge_idx];
                                            auto v_to_id = e_acc_1[edge_idx];
                                            const auto v_prev_from = v_prev[lid][v_from_id];
                                            const auto v_prev_to = v_prev[lid][v_to_id];


                                              if ((v_prev_from == SIR_INDIVIDUAL_S) && (v_prev_to == SIR_INDIVIDUAL_I))
                                              {
                                                  float p_I = p_I_acc[lid][connection_id];
                                                  Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                                                  auto &rng = rng_acc[lid];
                                                  bernoulli_I.p = p_I;
                                                  if (bernoulli_I(rng))
                                                  {
                                                    N_inf++;
                                                      v_next[lid][v_from_id] = SIR_INDIVIDUAL_I;
                                                      event_to_acc[lid][connection_id]++;
                                                  }
                                              }
                                              else if ((v_prev_from == SIR_INDIVIDUAL_I) && (v_prev_to == SIR_INDIVIDUAL_S))
                                              {
                                                  float p_I = p_I_acc[lid][ecm_acc[edge_idx]];
                                                  Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                                                  auto &rng = rng_acc[lid];
                                                  bernoulli_I.p = p_I;
                                                  if (bernoulli_I(rng))
                                                  {

                                                    N_inf++;
                                                      v_next[lid][v_to_id] = SIR_INDIVIDUAL_I;
                                                      event_from_acc[lid][connection_id]++;
                                                  }
                                              }

                                          }
                                          out << "N_inf for sim " << sim_id << ":\t" << N_inf << sycl::endl;
                                           });
                                    gr.parallel_for_work_item([&](sycl::h_item<1> it)
                                                                  {
                                        auto sim_id = it.get_global_id(0);
                                        auto lid = it.get_local_id(0);
                                        for(int v_idx = 0; v_idx < N_vertices; v_idx++)
                                        {
                                            v_glob_next[0][sim_id][v_idx] = v_next[lid][v_idx];

                                        }
                                        for(int i = 0; i < N_connections; i++)
                                        {
                                            event_from_acc_glob[0][sim_id][i] = event_from_acc[lid][i];
                                            event_to_acc_glob[0][sim_id][i] = event_to_acc[lid][i];
                                        }
                                        rng_acc_glob[sim_id] = rng_acc[lid];
                                                                  });


                                           }); });

    std::cout << "Timestep " << t << " done\n";
    return {inf_event};
}

#endif
