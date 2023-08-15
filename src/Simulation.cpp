
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

class init_kernel;
class infection_kernel;
class recovery__kernel;

Simulator::Simulator(sycl::queue &q,
                     cl::sycl::buffer<Static_RNG::default_rng> &rngs,
                     cl::sycl::buffer<SIR_State, 3> &trajectory,
                     cl::sycl::buffer<uint32_t, 3> &events_from,
                     cl::sycl::buffer<uint32_t, 3> &events_to,
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
                     const float p_I0, const float p_R0, const float p_R, const uint32_t Nt_alloc,
                     std::vector<sycl::event> alloc_events) : q(q), rngs(std::move(rngs)),
                                                              trajectory(std::move(trajectory)), events_from(std::move(events_from)),
                                                              events_to(std::move(events_to)), p_Is(std::move(p_Is)),
                                                              edge_from(std::move(edge_from)), edge_to(std::move(edge_to)),
                                                              ecm(std::move(ecm)), vcm(std::move(vcm)), vcm_vec(vcm_vec),
                                                              community_state(std::move(community_state)), ccm(ccm),
                                                              ccm_weights(ccm_weights), logger(std::move(logger)), output_dir(output_dir),
                                                              Nt(Nt), N_communities(N_communities), N_pop(N_pop), N_edges(N_edges),
                                                              N_threads(N_threads), seed(seed), N_connections(N_connections),
                                                              N_vertices(N_vertices), N_sims(N_sims), p_I0(p_I0), p_R0(p_R0), p_R(p_R),
                                                              alloc_events(alloc_events), max_infection_samples(1000), events_from_flat(N_sims * Nt * N_connections, 0),
                                                              events_to_flat(N_sims * Nt * N_connections, 0),
                                                              community_state_flat(N_sims * (Nt + 1) * N_communities), Nt_alloc(Nt_alloc)
{
    auto device = q.get_device();
    // get global mem size
    auto global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    // get max alloc size
    auto max_alloc_size = device.get_info<sycl::info::device::max_mem_alloc_size>();
    assert(this->byte_size() < global_mem_size);
    assert(this->byte_size() < max_alloc_size);
}

uint32_t Simulator::byte_size() const
{
    uint32_t byte_size = 0;
    byte_size += rngs.byte_size();
    byte_size += trajectory.byte_size();
    byte_size += events_from.byte_size();

    byte_size += events_to.byte_size();
    p_Is.byte_size();
    byte_size += edge_from.byte_size();
    byte_size += edge_to.byte_size();
    byte_size += ecm.byte_size();
    byte_size += vcm.byte_size();
    byte_size += community_state.byte_size();
    return byte_size;
}

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
    uint32_t N_vertices = this->N_vertices;
    auto [compute_range, wg_range] = get_work_group_ranges();
    uint32_t t_alloc = t % Nt_alloc;
    // auto event = dep_event[0];
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto rng_acc_glob = rngs.template get_access<sycl::access_mode::read_write>(h);
    auto v_glob_prev = sycl::accessor<SIR_State, 3, sycl::access_mode::read>(trajectory, h, sycl::range<3>(N_sims, 1, N_vertices), sycl::range<3>(0, t_alloc, 0));
    auto v_glob_next = sycl::accessor<SIR_State, 3, sycl::access_mode::write>(trajectory, h, sycl::range<3>(N_sims, 1, N_vertices), sycl::range<3>(0, t_alloc + 1, 0));
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
                v_prev[lid][v_idx] = v_glob_prev[gid][0][v_idx];
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
                v_glob_next[gid][0][v_idx] = v_next[lid][v_idx];
            }

        });

     }); });
    return {event};
}

std::vector<sycl::event> Simulator::enqueue()
{
    logger.log_start();
    logger.profile_f << "Enqueueing kernels..." << std::endl;
    // std::vector<sycl::event> dep_events = {};
    std::vector<sycl::event> dep_events = {this->initialize_vertices()};
    uint32_t t;
    for (t = 0; t < Nt; t++)
    {
        if ((!(t % Nt_alloc)) && (t != 0))
        {
            dep_events.push_back(read_reset_buffers(t, dep_events));
            q.wait();
        }
        dep_events = recover(t, dep_events);
        dep_events = infect(t, dep_events);
        q.wait();
    }

    dep_events = read_end_buffers(t % Nt_alloc, dep_events);
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
    uint32_t N_vertices = this->N_vertices;
    float p_I0 = this->p_I0;
    float p_R0 = this->p_R0;
    auto [compute_range, wg_range] = get_work_group_ranges();
    // sycl::event event;
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(alloc_events);
    // auto v_acc = sycl::accessor<SIR_State, 3, sycl::access_mode::write>(trajectory,h, sycl::range<3>(N_sims, 1, N_vertices), sycl::range<3>(0,0,0));
    auto v_acc = trajectory.template get_access<sycl::access_mode::write>(h);
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
                v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_I;
            } else if (bernoulli_R(rng)) {
                v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_R;
            } else {
                v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_S;
            }
            }
    }); }); });
    return event;
}

std::tuple<sycl::range<1>, sycl::range<1>> Simulator::get_work_group_ranges() const
{
    auto device = q.get_device();
    auto N_compute_units = device.template get_info<sycl::info::device::max_compute_units>();
    auto N_wg = std::min({this->get_work_group_size(), N_sims});
    if (N_sims > (N_compute_units * N_wg))
    {
        std::string err = "A maximum of " + std::to_string(N_compute_units * N_wg) + " simulations is supported concurrently on this device";
        throw std::runtime_error(err);
    }

    auto N_used_compute_units = static_cast<std::size_t>(std::ceil(static_cast<double>(N_sims) / static_cast<double>(N_wg)));

    return std::make_tuple(sycl::range<1>(N_used_compute_units), sycl::range<1>(N_wg));
}

std::vector<sycl::event> Simulator::infect(uint32_t t, std::vector<sycl::event> &dep_event)
{

    auto [compute_range, wg_range] = get_work_group_ranges();
    uint32_t N_connections = this->N_connections;
    uint32_t N_edges = this->N_edges;
    uint32_t N_vertices = this->N_vertices;
    uint32_t N_sims = this->N_sims;
    uint32_t t_alloc = t % Nt_alloc;
    assert(N_vertices == 200);
    // assert(N_sims == 8);
    // assert(N_connections = 3);
    assert(ecm.size() == N_edges);
    assert(vcm.size() == N_vertices);
    assert(p_Is.get_range()[0] == N_sims);
    assert(p_Is.get_range()[1] == Nt);
    assert(p_Is.get_range()[2] == N_connections);
    assert(rngs.size() == N_sims);
    assert(trajectory.get_range()[0] == N_sims);
    // assert(trajectory.get_range()[1] == (Nt + 1));
    assert(trajectory.get_range()[2] == N_vertices);
    assert(events_to.get_range()[0] == N_sims);
    // assert(events_to.get_range()[1] == Nt);
    assert(events_to.get_range()[2] == N_connections);
    assert(events_from.get_range()[0] == N_sims);
    // assert(events_from.get_range()[1] == Nt);
    assert(events_from.get_range()[2] == N_connections);
    auto inf_event = q.submit([&](sycl::handler &h)
                              {
        h.depends_on(dep_event);
        auto ecm_acc = ecm.template get_access<sycl::access::mode::read>(h);
        auto p_I_acc_glob = sycl::accessor<float, 3, sycl::access::mode::read>(p_Is, h, sycl::range<3>(N_sims, 1, N_connections), sycl::range<3>(0, t, 0));
        auto rng_acc_glob = rngs.template get_access<sycl::access::mode::read_write>(h);
        sycl::accessor<SIR_State, 3, sycl::access_mode::write> v_glob_next(trajectory, h, sycl::range<3>(N_sims, 1, N_vertices), sycl::range<3>(0, t_alloc + 1, 0));
        auto e_acc_0 = edge_to.template get_access<sycl::access::mode::read>(h);
        auto e_acc_1 = edge_from.template get_access<sycl::access::mode::read>(h);
        auto event_to_acc_glob = sycl::accessor<uint32_t, 3, sycl::access::mode::write>(events_to, h, sycl::range<3>(N_sims, 1, N_connections), sycl::range<3>(0, t_alloc, 0));
        auto event_from_acc_glob = sycl::accessor<uint32_t, 3, sycl::access::mode::write>(events_from, h, sycl::range<3>(N_sims, 1, N_connections), sycl::range<3>(0, t_alloc, 0));
        sycl::local_accessor<Static_RNG::default_rng> rng_acc(wg_range, h);
        sycl::local_accessor<SIR_State, 2> v_prev(sycl::range<2>(wg_range[0], N_vertices), h);
        sycl::local_accessor<SIR_State, 2> v_next(sycl::range<2>(wg_range[0], N_vertices), h);
        sycl::local_accessor<float, 2> p_I_acc(sycl::range<2>(wg_range[0], N_connections), h);
        sycl::local_accessor<uint32_t, 2> event_to_acc(sycl::range<2>(wg_range[0], N_connections), h);
        sycl::local_accessor<uint32_t, 2> event_from_acc(sycl::range<2>(wg_range[0], N_connections), h);
        auto local_mem_size_used = rng_acc.byte_size() + v_prev.byte_size() + v_next.byte_size() + p_I_acc.byte_size() + event_to_acc.byte_size() + event_from_acc.byte_size();
        auto lms = this->local_mem_size();
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
                                            v_prev[lid][v_idx] = v_glob_next[sim_id][0][v_idx];
                                            v_next[lid][v_idx] = v_glob_next[sim_id][0][v_idx];
                                        }
                                        for(int i = 0; i < N_connections; i++)
                                        {
                                            p_I_acc[lid][i] = p_I_acc_glob[sim_id][0][i];
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
                                                    // if (sim_id == 0)
                                                    // {
                                                    //     out << "infection from " << e_acc_0[edge_idx] << " to " << e_acc_1[edge_idx] << sycl::endl;
                                                    // }
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

                                                    // if (sim_id == 0)
                                                    // {
                                                    //     out << "infection from " << e_acc_1[edge_idx] << " to " << e_acc_0[edge_idx] << sycl::endl;
                                                    // }
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
                                            v_glob_next[sim_id][0][v_idx] = v_next[lid][v_idx];

                                        }
                                        for(int i = 0; i < N_connections; i++)
                                        {
                                            event_from_acc_glob[sim_id][0][i] = event_from_acc[lid][i];
                                            event_to_acc_glob[sim_id][0][i] = event_to_acc[lid][i];
                                        }
                                        rng_acc_glob[sim_id] = rng_acc[lid];
                                                                  });


                                           }); });

    // const auto vertex_trajectory = vector_remap(read_buffer<SIR_State, 3>(trajectory, q, ev), N_sims, Nt + 1, N_vertices);
    // if (t > 0)
    // {
    //     std::for_each(vertex_trajectory.begin(), vertex_trajectory.end(), [&](const auto &v)
    //                   {
    //     for (int i = 0; i < t; i++)
    //     {
    //         std::vector<std::pair<SIR_State, SIR_State>> comp(N_vertices);
    //         for (int v_idx = 0; v_idx < N_vertices; v_idx++)
    //         {
    //             comp[v_idx] = std::make_pair(v[i][v_idx], v[i + 1][v_idx]);
    //         }
    //         assert(!std::all_of(comp.begin(), comp.end(), [&](const auto &vp)
    //                             { return vp.first != vp.second; }));
    //     } });
    // }
    std::cout << "Timestep " << t << " done\n";
    return {inf_event};
}

std::size_t Simulator::local_mem_size() const
{
    auto device = q.get_device();
    auto max_local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    return max_local_mem_size;
}

sycl::event Simulator::accumulate_community_state(std::vector<sycl::event> &events, uint32_t t_max)
{
    uint32_t t_end = std::min({this->Nt_alloc, t_max});
    uint32_t N_vertices = this->N_vertices;
    auto [compute_range, wg_range] = get_work_group_ranges();
    return q.submit([&](sycl::handler &h)
                    {
                h.depends_on(events);
        auto v_acc = trajectory.template get_access<sycl::access::mode::read>(h);
        auto state_acc = community_state.template get_access<sycl::access::mode::read_write>(h);
        auto vcm_acc = vcm.template get_access<sycl::access::mode::read>(h);
        h.parallel_for_work_group(compute_range, wg_range, [=](sycl::group<1> gr)
        {
            gr.parallel_for_work_item([&](sycl::h_item<1> it)
            {
                auto sim_id = it.get_global_id(0);
                auto lid = it.get_local_id(0);
                for(int t = 0; t < t_end + 1; t++)
                {
                    for(int v_idx = 0; v_idx < N_vertices; v_idx++)
                    {
                        auto c_idx = vcm_acc[v_idx];
                        auto v_state = v_acc[sim_id][t][v_idx];
                        state_acc[sim_id][t][c_idx][v_state]++;
                    }
                }
            });
        }); });
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

std::vector<sycl::event> Simulator::read_end_buffers(uint32_t t, std::vector<sycl::event> &dep_events)
{
    std::vector<sycl::event> read_events(4);
    auto t_end = t % Nt_alloc + 1;
    read_events[0] = accumulate_community_state(dep_events, t_end);
    uint32_t event_offset = (t - Nt_alloc) * N_sims * N_connections;
    // return sycl::event{};
    read_events[0] = q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_events);
        auto acc = sycl::accessor<SIR_State, 3, sycl::access::mode::read>(trajectory, h, sycl::range<3>(N_sims, t_end+1, N_vertices), sycl::range<3>(0,0,0));
        h.copy(acc, &community_state_flat[event_offset]);
    });

    read_events[1] = q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_events);
        auto acc = sycl::accessor<uint32_t, 3, sycl::access::mode::read>(events_from, h, sycl::range<3>(N_sims, t_end, N_connections), sycl::range<3>(0,0,0));
        h.copy(acc, &events_from_flat[event_offset]);
    });

    read_events[2] = q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_events);
        auto acc = sycl::accessor<uint32_t, 3, sycl::access::mode::read>(events_to, h, sycl::range<3>(N_sims, t_end, N_connections), sycl::range<3>(0,0,0));
        h.copy(acc, &events_to_flat[event_offset]);
    });
    read_events.insert(read_events.end(), dep_events.begin(), dep_events.end());
    return read_events;
}

sycl::event Simulator::read_reset_buffers(uint32_t t, std::vector<sycl::event> &dep_events)
{
    std::vector<sycl::event> read_events(4);
    read_events[0] = accumulate_community_state(dep_events, Nt_alloc);
    uint32_t event_offset = (t - Nt_alloc) * N_sims * N_connections;
    auto t_end = Nt % Nt_alloc;
    uint32_t Nt_alloc = this->Nt_alloc;
    // return sycl::event{};
    read_events[1] = read_buffer<State_t, 3>(community_state, q, &community_state_flat[event_offset], dep_events);
    read_events[2] = read_buffer<uint32_t, 3>(events_from, q, &events_from_flat[event_offset], dep_events);
    read_events[3] = read_buffer<uint32_t, 3>(events_to, q, &events_to_flat[event_offset], dep_events);
    read_events.insert(read_events.end(), dep_events.begin(), dep_events.end());
    return q.submit([&](sycl::handler &h)
                    {
        h.depends_on(read_events);
        sycl::accessor<SIR_State, 3, sycl::access::mode::read> v_end(trajectory, h, sycl::range<3>(N_sims, 1, N_vertices), sycl::range<3>(0,t_end + 1,0));
        sycl::accessor<SIR_State, 3, sycl::access::mode::write> v_start(trajectory, h, sycl::range<3>(N_sims, 1, N_vertices), sycl::range<3>(0,0,0));
        h.copy(v_end, v_start); });
}

void Simulator::write_to_files(std::vector<sycl::event> &dep_events)
{
    logger.profile_events(dep_events);

    logger.profile_f << "Sampling infections\n";
    logger.log_start();
    auto community_state = vector_remap(community_state_flat, N_sims, Nt + 1, N_communities);
    auto events_from_vec = vector_remap(events_from_flat, N_sims, Nt, N_connections);
    auto events_to_vec = vector_remap(events_to_flat, N_sims, Nt, N_connections);
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
    uint32_t N_vertices = vcm_init.size();

    std::vector<uint32_t> edge_from_init(edge_list.size());
    std::vector<uint32_t> edge_to_init(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), edge_from_init.begin(), [](auto &e)
                   { return e.first; });
    std::transform(edge_list.begin(), edge_list.end(), edge_to_init.begin(), [](auto &e)
                   { return e.second; });
    std::vector<State_t> community_state_init(p.N_communities * p.N_sims * (p.Nt_alloc + 1), {0, 0, 0});
    std::vector<SIR_State> traj_init(p.N_sims * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
    std::vector<uint32_t> event_init(p.N_sims * N_connections * p.Nt_alloc, 0);
    auto seeds = generate_seeds(p.N_sims, p.seed);
    std::vector<Static_RNG::default_rng> rng_init;
    rng_init.reserve(p.N_sims);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](auto &seed)
                   { return Static_RNG::default_rng(seed); });

    auto p_Is = sycl::buffer<float, 3>(sycl::range<3>(p.N_sims, p.Nt, N_connections));
    auto edge_from = sycl::buffer<uint32_t>(sycl::range<1>(edge_from_init.size()));
    auto edge_to = sycl::buffer<uint32_t>(sycl::range<1>(edge_to_init.size()));
    auto ecm = sycl::buffer<uint32_t>(sycl::range<1>(ecm_init.size()));
    auto vcm = sycl::buffer<uint32_t>(sycl::range<1>(vcm_init.size()));
    auto community_state = sycl::buffer<State_t, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc + 1, p.N_communities));
    auto trajectory = sycl::buffer<SIR_State, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc + 1, N_vertices));
    auto events_from = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc, N_connections));
    auto events_to = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc, N_connections));
    auto rngs = sycl::buffer<Static_RNG::default_rng>(sycl::range<1>(rng_init.size()));

    std::vector<sycl::event> alloc_events(10);
    alloc_events[0] = initialize_device_buffer<float, 3>(q, p_Is_init, p_Is);
    alloc_events[1] = initialize_device_buffer<uint32_t, 1>(q, edge_from_init, edge_from);
    alloc_events[2] = initialize_device_buffer<uint32_t, 1>(q, edge_to_init, edge_to);
    alloc_events[3] = initialize_device_buffer<uint32_t, 1>(q, ecm_init, ecm);
    alloc_events[4] = initialize_device_buffer<uint32_t, 1>(q, vcm_init, vcm);
    alloc_events[5] = initialize_device_buffer<State_t, 3>(q, community_state_init, community_state);
    alloc_events[6] = initialize_device_buffer<SIR_State, 3>(q, traj_init, trajectory);
    alloc_events[7] = initialize_device_buffer<uint32_t, 3>(q, event_init, events_from);
    alloc_events[8] = initialize_device_buffer<uint32_t, 3>(q, event_init, events_to);
    alloc_events[9] = initialize_device_buffer<Static_RNG::default_rng, 1>(q, rng_init, rngs);

    auto ccm = complete_ccm(p.N_communities);
    auto ccm_weights = ccm_weights_from_ecm(ecm_init);

    return Simulator(q, rngs, trajectory, events_from, events_to, p_Is,
                     edge_from, edge_to, ecm,
                     vcm, vcm_init, community_state, ccm, ccm_weights,
                     Simulation_Logger(p.output_dir), p.output_dir,
                     p.Nt, p.N_communities,
                     p.N_pop, edge_list.size(), p.N_threads, p.seed,
                     N_connections, p.N_pop * p.N_communities, p.N_sims, p.p_I0, p.p_R0, p.p_R, p.Nt_alloc, alloc_events);
}
