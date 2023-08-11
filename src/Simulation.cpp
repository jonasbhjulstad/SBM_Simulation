
#include <Sycl_Graph/Simulation.hpp>

#include <Sycl_Graph/Buffer_Utils.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Profiling.hpp>
#include <Sycl_Graph/SIR_Infection_Sampling.hpp>
#include <algorithm>
#include <cmath>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

Simulator::Simulator(sycl::queue &q,
                     cl::sycl::buffer<Static_RNG::default_rng, 2> &rngs,
                     cl::sycl::buffer<SIR_State, 3> &trajectory,
                     cl::sycl::buffer<uint32_t, 3> &events_from,
                     cl::sycl::buffer<uint32_t, 3> &events_to,
                     cl::sycl::buffer<uint8_t, 2> &infections_from,
                     cl::sycl::buffer<uint8_t, 2> &infections_to,
                     cl::sycl::buffer<float, 3> &p_Is,
                     cl::sycl::buffer<uint32_t> &edge_from,
                     cl::sycl::buffer<uint32_t> &edge_to,
                     cl::sycl::buffer<uint32_t> &ecm,
                     cl::sycl::buffer<uint32_t> &vcm,
                     const std::vector<uint32_t> &vcm_vec,
                     cl::sycl::buffer<State_t, 3> &community_state,
                     const std::vector<std::pair<uint32_t, uint32_t>> &ccm,
                     const std::vector<uint32_t> &ccm_weights,
                     Simulation_Logger logger,
                     const std::string &output_dir,
                     const uint32_t Nt,
                     const uint32_t N_communities,
                     const uint32_t N_pop,
                     const uint32_t N_edges,
                     const uint32_t N_threads,
                     const uint32_t seed,
                     const uint32_t N_connections,
                     const uint32_t N_vertices,
                     const uint32_t N_sims,
                     const float p_I0, const float p_R0, const float p_R,
                     std::vector<sycl::event> alloc_events) : q(q), rngs(std::move(rngs)),
                                                              trajectory(std::move(trajectory)), events_from(std::move(events_from)),
                                                              events_to(std::move(events_to)), infections_from(std::move(infections_from)),
                                                              infections_to(std::move(infections_to)), p_Is(std::move(p_Is)),
                                                              edge_from(std::move(edge_from)), edge_to(std::move(edge_to)),
                                                              ecm(std::move(ecm)), vcm(std::move(vcm)), vcm_vec(vcm_vec),
                                                              community_state(std::move(community_state)), ccm(ccm),
                                                              ccm_weights(ccm_weights), logger(std::move(logger)), output_dir(output_dir),
                                                              Nt(Nt), N_communities(N_communities), N_pop(N_pop), N_edges(N_edges),
                                                              N_threads(N_threads), seed(seed), N_connections(N_connections),
                                                              N_vertices(N_vertices), N_sims(N_sims), p_I0(p_I0), p_R0(p_R0), p_R(p_R),
                                                              alloc_events(alloc_events), max_infection_samples(1000) {}

uint32_t Simulator::N_vertex_per_thread() const
{
    return static_cast<uint32_t>(std::ceil(static_cast<double>(N_vertices) / static_cast<double>(N_threads)));
}

uint32_t Simulator::N_edge_per_thread() const
{
    return static_cast<uint32_t>(std::ceil(static_cast<double>(N_edges) / static_cast<double>(N_threads)));
}

std::vector<sycl::event> Simulator::recover(uint32_t t, std::vector<sycl::event> &dep_event)
{
    const float p_R = this->p_R;
    uint32_t Nvpt = N_vertex_per_thread();
    uint32_t N_vertices = this->N_vertices;
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto rng_acc = rngs.template get_access<sycl::access_mode::read_write>(h);
    sycl::accessor<SIR_State, 3, sycl::access_mode::read_write> v_acc(trajectory, h, sycl::range<3>(N_sims, 2, N_vertices), sycl::range<3>(0, t, 0));
    h.parallel_for(get_nd_range(), [=](sycl::nd_item<1> it)
    {
        auto sim_id = it.get_global_id()[0];
        auto local_id = it.get_local_id()[0];
        auto& rng = rng_acc[sim_id][local_id];
        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
        for(int i = 0; i < Nvpt; i++)
        {
            auto v_idx = Nvpt*local_id + i;
            if(v_idx >= N_vertices)
                break;
      auto state_prev = v_acc[sim_id][0][v_idx];
      if (state_prev == SIR_INDIVIDUAL_I) {
        if (bernoulli_R(rng)) {
          v_acc[sim_id][1][v_idx] = SIR_INDIVIDUAL_R;
        }
      }
        }
    }); });
    return {event};
}

std::vector<sycl::event> Simulator::enqueue()
{
    logger.log_start();
    logger.profile_f << "Enqueueing kernels..." << std::endl;
    std::vector<sycl::event> dep_events = {this->initialize_vertices()};
    for (int t = 0; t < Nt; t++)
    {
        dep_events = recover(t, dep_events);
        dep_events = infect(t, dep_events);
    }
    logger.log_end();

    return dep_events;
}

void Simulator::run()
{
    auto dep_events = enqueue();
    write_to_files(dep_events);
}

sycl::event Simulator::initialize_vertices()
{
    uint32_t N_vertex_per_thread = this->N_vertex_per_thread();
    uint32_t N_vertices = this->N_vertices;
    float p_I0 = this->p_I0;
    float p_R0 = this->p_R0;
    return q.submit([&](sycl::handler &h)
                    {
    h.depends_on(alloc_events);
    auto v_acc = sycl::accessor<SIR_State, 3, sycl::access_mode::write>(trajectory,h, sycl::range<3>(N_sims, 1, N_vertices), sycl::range<3>(0,0,0));
    auto rng_acc =
        rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(get_nd_range(), [=](sycl::nd_item<1> it) {
            auto sim_id = it.get_global_id()[0];
            auto local_id = it.get_local_id()[0];
            for(int i = 0; i < N_vertex_per_thread; i++)
            {
            auto vertex_idx = N_vertex_per_thread*local_id + i;
            if (vertex_idx >= N_vertices)
                break;
            auto& rng = rng_acc[sim_id][local_id];
            Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
            Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);
            if (bernoulli_I(rng)) {
                v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_I;
            } else if (bernoulli_R(rng)) {
                v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_R;
            } else {
                v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_S;
            }
            }
    }); });
}

std::vector<sycl::event> Simulator::infect(uint32_t t, std::vector<sycl::event> &dep_event)
{
    uint32_t N_connections = this->N_connections;
    uint32_t N_edges = this->N_edges;
    uint32_t Nept = N_edge_per_thread();
    auto range = get_nd_range();
    auto inf_event = q.submit([&](sycl::handler &h)
                              {
        h.depends_on(dep_event);
        auto ecm_acc = ecm.template get_access<sycl::access::mode::read>(h);
        auto p_I_acc = sycl::accessor<float, 3, sycl::access::mode::read>(p_Is, h, sycl::range<3>(N_sims, 1, N_connections), sycl::range<3>(0, t, 0));
        auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
        sycl::accessor<SIR_State, 3, sycl::access_mode::read_write> v_acc(trajectory, h, sycl::range<3>(N_sims, 2, N_vertices), sycl::range<3>(0, t, 0));

        auto e_acc_0 = edge_to.template get_access<sycl::access::mode::read>(h);
        auto e_acc_1 = edge_from.template get_access<sycl::access::mode::read>(h);
        auto inf_to_acc = infections_to.template get_access<sycl::access::mode::read_write>(h);
        auto inf_from_acc = infections_from.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(range, [=](sycl::nd_item<1> it)
                       {

            auto sim_id = it.get_global_id()[0];
            auto local_id = it.get_local_id()[0];
                for (int i = 0; i < Nept; i++)
                {
                    auto edge_idx = Nept*local_id + i;
                    if (edge_idx >= N_edges)
                        break;
                    if ((v_acc[sim_id][0][e_acc_1[edge_idx]] == SIR_INDIVIDUAL_S) && (v_acc[sim_id][0][e_acc_0[edge_idx]] == SIR_INDIVIDUAL_I))
                    {
                        float p_I = p_I_acc[sim_id][0][ecm_acc[edge_idx]];
                        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                        auto& rng = rng_acc[sim_id][local_id];
                        bernoulli_I.p = p_I;
                        if (bernoulli_I(rng))
                        {
                            v_acc[sim_id][1][e_acc_1[edge_idx]] = SIR_INDIVIDUAL_I;
                            inf_to_acc[sim_id][edge_idx] = 1;
                        }
                    }
                    else if ((v_acc[sim_id][0][e_acc_1[edge_idx]] == SIR_INDIVIDUAL_I) && (v_acc[sim_id][0][e_acc_0[edge_idx]] == SIR_INDIVIDUAL_S))
                    {
                        float p_I = p_I_acc[sim_id][0][ecm_acc[edge_idx]];
                        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                        auto& rng = rng_acc[sim_id][local_id];
                        bernoulli_I.p = p_I;
                        if (bernoulli_I(rng))
                        {
                            v_acc[sim_id][1][e_acc_1[edge_idx]] = SIR_INDIVIDUAL_I;
                            inf_from_acc[sim_id][edge_idx] = 1;
                        }
                    }
                } }); });
    auto acc_event = q.submit([&](sycl::handler &h)
                              {
        h.depends_on(inf_event);
        auto inf_to_acc = infections_to.template get_access<sycl::access::mode::read>(h);
        auto inf_from_acc = infections_from.template get_access<sycl::access::mode::read>(h);
        auto event_to_acc = sycl::accessor<uint32_t, 3, sycl::access::mode::read_write>(events_to, h, sycl::range<3>(N_sims, 1, N_connections), sycl::range<3>(N_sims, t, 0));
        auto event_from_acc = sycl::accessor<uint32_t, 3, sycl::access::mode::read_write>(events_to, h, sycl::range<3>(N_sims, 1, N_connections), sycl::range<3>(N_sims, t, 0));
        auto ecm_acc = ecm.template get_access<sycl::access::mode::read>(h);
        h.parallel_for(N_sims, [=](sycl::item<1> idx)
                       {
                        for(int i = 0; i < N_connections; i++)
                        {
                            event_to_acc[idx][0][i] = 0;
                            event_from_acc[idx][0][i] = 0;
                        }
                        for(int i = 0; i < N_edges; i++)
                        {
                            event_to_acc[idx][0][ecm_acc[i]] += inf_to_acc[idx][i];
                            event_from_acc[idx][0][ecm_acc[i]] += inf_from_acc[idx][i];
                        }
}); });

    return {acc_event};
}

sycl::nd_range<1> Simulator::get_nd_range() const
{
    uint32_t sqrt_N_sims = std::sqrt(N_sims);
    assert(sqrt_N_sims * sqrt_N_sims == N_sims);
    return sycl::nd_range<1>(sycl::range<1>(N_sims), sycl::range<1>(get_work_group_size()));
}

std::vector<std::vector<std::vector<State_t>>> Simulator::accumulate_community_state(std::vector<sycl::event> &events, sycl::event &res_event)
{
    logger.profile_f << "State accumulation\n";
    logger.profile_events(events);
    const auto vertex_trajectory = vector_remap(read_buffer<SIR_State, 3>(trajectory, q, res_event), N_sims, Nt + 1, N_vertices);
    std::vector<std::vector<std::vector<State_t>>> community_state(N_sims, std::vector<std::vector<State_t>>(Nt + 1, std::vector<State_t>(N_communities)));
    std::transform(std::execution::par_unseq, vertex_trajectory.begin(), vertex_trajectory.end(), community_state.begin(),
                   [&](const auto &sim_traj)
                   {
                       std::vector<std::vector<State_t>> sim_community_state(Nt + 1, std::vector<State_t>(N_communities));
                       std::transform(std::execution::par_unseq, sim_traj.begin(), sim_traj.end(), sim_community_state.begin(), [&](const auto &vertex_state_t)
                                      {
                            std::vector<State_t> community_state(N_communities, {0, 0, 0});
                            for (int i = 0; i < N_vertices; i++)
                            {
                                community_state[vcm_vec[i]][vertex_state_t[i]]++;
                            }
                            return community_state; });
                       return sim_community_state;
                   });
    return community_state;
}

std::vector<std::vector<std::vector<uint32_t>>> Simulator::sample_from_connection_events(const std::vector<std::vector<std::vector<State_t>>> &community_state,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &from_events,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &to_events)
{
    std::vector<std::vector<std::vector<uint32_t>>> infections = std::vector<std::vector<std::vector<uint32_t>>>(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections)));

    std::vector<std::tuple<std::vector<std::vector<State_t>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>>> sim_data(N_sims);

    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        sim_data[sim_idx] = std::make_tuple(community_state[sim_idx], from_events[sim_idx], to_events[sim_idx]);
    }
    std::transform(std::execution::par_unseq, sim_data.begin(), sim_data.end(), infections.begin(), [&](const auto &sim_d)
                   {
        const auto& sim_community_state = std::get<0>(sim_d);
        const auto& sim_from_events = std::get<1>(sim_d);
        const auto& sim_to_events = std::get<2>(sim_d);
        return sample_infections(sim_community_state, sim_from_events, sim_to_events, ccm, ccm_weights, seed, max_infection_samples); });

    return infections;
}

void Simulator::write_to_files(std::vector<sycl::event> &dep_events)
{
    std::vector<sycl::event> res_events(3);
    auto community_state = accumulate_community_state(dep_events, res_events[0]);
    auto events_from_vec = vector_remap(read_buffer<uint32_t, 3>(events_from, q, res_events[1]), N_sims, Nt, N_connections);
    auto events_to_vec = vector_remap(read_buffer<uint32_t, 3>(events_to, q, res_events[2]), N_sims, Nt, N_connections);
    logger.profile_events(res_events);
    logger.profile_f << "Sampling infections\n";
    logger.log_start();
    auto infections = sample_from_connection_events(community_state, events_from_vec, events_to_vec);
    logger.log_end();
    logger.profile_f << "Writing to files\n";

    logger.log_start();
    std::filesystem::create_directories(output_dir);
    std::ofstream community_traj_f;
    std::ofstream connection_events_f;
    std::ofstream connection_infections_f;

    for (int sim_id = 0; sim_id < N_sims; sim_id++)
    {

        community_traj_f.open(output_dir + "community_trajectory_" +
                              std::to_string(sim_id) + ".csv");
        std::for_each(community_state[sim_id].begin(), community_state[sim_id].end(),
                      [&](auto &community_trajectory_i)
                      {
                          linewrite(community_traj_f, community_trajectory_i);
                      });
        community_traj_f.close();

        connection_events_f.open(output_dir + "connection_events_" +
                                 std::to_string(sim_id) + ".csv");
        for (int t = 0; t < Nt; t++)
        {
            std::vector<std::vector<uint32_t>> connection_pair = {{events_from_vec[sim_id][t], events_to_vec[sim_id][t]}};
            auto connections = merge_vectors(connection_pair);
            linewrite(connection_events_f, connections);
        }
        connection_events_f.close();
        connection_infections_f.open(output_dir + "connection_infections_" +
                                     std::to_string(sim_id) + ".csv");

        std::for_each(infections[sim_id].begin(), infections[sim_id].end(), [&](auto &infections_i)
                      { linewrite(connection_infections_f, infections_i); });
    }
    logger.log_end();
}
uint32_t Simulator::get_work_group_size() const
{
    return q.get_device().get_info<sycl::info::device::max_work_group_size>();
}

Simulator make_SIR_simulation(sycl::queue &q, const Sim_Param &p, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &vcm_init, const float p_I_min, const float p_I_max)
{
    auto ecm_init = ecm_from_vcm(edge_list, vcm_init);

    uint32_t N_connections = std::max_element(ecm_init.begin(), ecm_init.end())[0] + 1;

    auto p_Is = generate_floats(p.N_sims * p.Nt * N_connections, p_I_min, p_I_max, p.seed);
    return make_SIR_simulation(q, p, edge_list, vcm_init, p_Is);
}

Simulator make_SIR_simulation(sycl::queue &q, const Sim_Param &p, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &vcm_init, const std::vector<float> &p_Is_init)
{
    auto ecm_init = ecm_from_vcm(edge_list, vcm_init);

    uint32_t N_connections = std::max_element(ecm_init.begin(), ecm_init.end())[0] + 1;

    // Initialize Common Buffers
    // get work group size
    auto device = q.get_device();
    uint32_t N_wg = device.get_info<sycl::info::device::max_work_group_size>();
    sycl::event rng_event;
    auto rngs = generate_rngs(q, sycl::range<2>(p.N_sims, N_wg), p.seed, rng_event);
    uint32_t N_vertices = vcm_init.size();
    auto trajectory = cl::sycl::buffer<SIR_State, 3>(sycl::range<3>(p.N_sims, p.Nt + 1, N_vertices));
    auto events_from = cl::sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, p.Nt, N_connections));
    auto events_to = cl::sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, p.Nt, N_connections));
    auto infections_from = cl::sycl::buffer<uint8_t, 2>(sycl::range<2>(p.N_sims, N_vertices));
    auto infections_to = cl::sycl::buffer<uint8_t, 2>(sycl::range<2>(p.N_sims, N_vertices));
    std::vector<sycl::event> alloc_events(6);
    auto p_Is = create_device_buffer<float, 3>(q, p_Is_init, sycl::range<3>(p.N_sims, p.Nt, N_connections), alloc_events[0]);

    std::vector<uint32_t> edge_from_init(edge_list.size());
    std::vector<uint32_t> edge_to_init(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), edge_from_init.begin(), [](auto &e)
                   { return e.first; });
    std::transform(edge_list.begin(), edge_list.end(), edge_to_init.begin(), [](auto &e)
                   { return e.second; });

    auto edge_from = create_device_buffer<uint32_t>(q, edge_from_init, alloc_events[1]);
    auto edge_to = create_device_buffer<uint32_t>(q, edge_to_init, alloc_events[2]);

    auto ecm = create_device_buffer<uint32_t>(q, ecm_init, alloc_events[3]);
    auto vcm = create_device_buffer<uint32_t>(q, vcm_init, alloc_events[4]);
    std::vector<State_t> community_state_init(p.N_communities * p.N_sims * (p.Nt + 1), {0, 0, 0});
    auto community_state = create_device_buffer<State_t, 3>(q, community_state_init, sycl::range<3>(p.N_sims, p.Nt + 1, p.N_communities), alloc_events[5]);

    auto ccm = complete_ccm(p.N_communities);
    auto ccm_weights = ccm_weights_from_ecm(ecm_init);

    return Simulator(q, rngs, trajectory, events_from, events_to,
                     infections_from, infections_to, p_Is,
                     edge_from, edge_to, ecm,
                     vcm, vcm_init, community_state, ccm, ccm_weights,
                     Simulation_Logger(p.output_dir), p.output_dir,
                     p.Nt, p.N_communities,
                     p.N_pop, edge_list.size(), p.N_threads, p.seed,
                     N_connections, p.N_pop * p.N_communities, p.N_sims, p.p_I0, p.p_R0, p.p_R, alloc_events);
}
