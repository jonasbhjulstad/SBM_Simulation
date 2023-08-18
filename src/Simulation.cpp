
#include <Sycl_Graph/Buffer_Utils.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Profiling.hpp>
#include <Sycl_Graph/SIR_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation.hpp>
#include <Sycl_Graph/Community_State.hpp>
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
                                                              alloc_events(alloc_events), max_infection_samples(1000), events_from_flat(N_sims * Nt_alloc * N_connections, 0),
                                                              events_to_flat(N_sims * Nt_alloc * N_connections, 0),
                                                              community_state_flat(N_sims * (Nt_alloc) * N_communities), Nt_alloc(Nt_alloc),
                                                              community_timeseries(std::vector<std::vector<std::vector<State_t>>>(N_sims, std::vector<std::vector<State_t>>(Nt+1, std::vector<State_t>(N_communities, {0,0,0})))),
                                                            events_from_timeseries(std::vector<std::vector<std::vector<uint32_t>>>(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0)))),
                                                            events_to_timeseries(std::vector<std::vector<std::vector<uint32_t>>>(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0))))
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

std::vector<sycl::event> Simulator::enqueue()
{
    logger.log_start();
    logger.profile_f << "Enqueueing kernels..." << std::endl;
    // std::vector<sycl::event> dep_events = {};
    std::vector<sycl::event> dep_events = {this->initialize_vertices()};
    uint32_t t;
    auto [compute_range, wg_range] = get_work_group_ranges();
    q.wait();
    for (t = 0; t < Nt; t++)
    {
        if ((!(t % Nt_alloc)) && (t != 0))
        {
            dep_events = (read_reset_buffers(t, dep_events));
        }
        dep_events = recover(t, dep_events);
        dep_events = infect(t, dep_events);
        q.wait();
    }

    dep_events = read_reset_buffers(t, dep_events);
    logger.log_end();

    return dep_events;
}

void Simulator::run()
{
    auto dep_events = enqueue();
    write_to_files(dep_events);
    q.wait();
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

std::size_t Simulator::local_mem_size() const
{
    auto device = q.get_device();
    auto max_local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    return max_local_mem_size;
}

std::vector<std::vector<std::vector<uint32_t>>> Simulator::sample_from_connection_events(const std::vector<std::vector<std::vector<State_t>>> &community_state,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &from_events,
                                                                                         const std::vector<std::vector<std::vector<uint32_t>>> &to_events)
{
    std::vector<std::vector<std::vector<uint32_t>>> infections = std::vector<std::vector<std::vector<uint32_t>>>(N_sims, std::vector<std::vector<uint32_t>>(N_connections, std::vector<uint32_t>(Nt)));

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

auto merge(const std::vector<std::vector<std::vector<uint32_t>>>& t0, const std::vector<std::vector<std::vector<uint32_t>>>& t1)
{
    auto N_sims = t0.size();
    auto Nt = t0[0].size();
    auto N_cols = t0[0][0].size();

    std::vector<std::vector<std::vector<uint32_t>>> merged(t0.size(), std::vector<std::vector<uint32_t>>(t0[0].size(), std::vector<uint32_t>(2*t0[0][0].size())));
    for(int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        for(int t = 0; t < Nt; t++)
        {
            for(int col = 0; col < N_cols; col++)
            {
                merged[sim_idx][t][2*col] = t0[sim_idx][t][col];
                merged[sim_idx][t][2*col+1] = t1[sim_idx][t][col];
            }
        }
    }
    return merged;

}

void events_to_file(const std::vector<std::vector<std::vector<uint32_t>>>& e_to, const std::vector<std::vector<std::vector<uint32_t>>>& e_from, const std::string& abs_fname)
{
    auto merged = merge(e_to, e_from);
    std::ofstream f;
    std::for_each(merged.begin(), merged.end(), [&, n = 0](const auto& sim_ts) mutable
                  {
        f.open(abs_fname + "_" + std::to_string(n) + ".csv");
        std::for_each(sim_ts.begin(), sim_ts.end(), [&](const auto& v)
                      {
            std::for_each(v.begin(), v.end(), [&](const auto& e)
                          {
                f << e << ",";
            });
            f << "\n";
        });
        f.close();
        n++;
    });
}
void timeseries_to_file(const std::vector<std::vector<std::vector<State_t>>>& ts, const std::string& abs_fname)
{
    std::ofstream f;
    std::for_each(ts.begin(), ts.end(), [&, n = 0](const auto& sim_ts) mutable
                  {
        f.open(abs_fname + "_" + std::to_string(n) + ".csv");
        std::for_each(sim_ts.begin(), sim_ts.end(), [&](const auto& v)
                      {
            std::for_each(v.begin(), v.end(), [&](const auto& e)
                          {
                f << e[0] << "," << e[1] << "," << e[2] << ",";
            });
            f << "\n";
        });
        f.close();
        n++;
    });
}



void Simulator::write_to_files(std::vector<sycl::event> &dep_events)
{
    logger.profile_events(dep_events);

    timeseries_to_file(community_timeseries, output_dir + "/community_trajectory");
    events_to_file(events_to_timeseries, events_from_timeseries, output_dir + "/connection_events");

    logger.profile_f << "Sampling infections\n";
    logger.log_start();
    auto infections = sample_from_connection_events(community_timeseries, events_from_timeseries, events_to_timeseries);
    logger.log_end();
    logger.profile_f << "Writing to files\n";

    logger.log_start();
    std::filesystem::create_directories(output_dir);



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

    auto p_Is = sycl::buffer<float, 3>(sycl::range<3>(p.Nt, p.N_sims, N_connections));
    auto edge_from = sycl::buffer<uint32_t>(sycl::range<1>(edge_from_init.size()));
    auto edge_to = sycl::buffer<uint32_t>(sycl::range<1>(edge_to_init.size()));
    auto ecm = sycl::buffer<uint32_t>(sycl::range<1>(ecm_init.size()));
    auto vcm = sycl::buffer<uint32_t>(sycl::range<1>(vcm_init.size()));
    auto community_state = sycl::buffer<State_t, 3>(sycl::range<3>(p.Nt_alloc + 1, p.N_sims, p.N_communities));
    auto trajectory = sycl::buffer<SIR_State, 3>(sycl::range<3>(p.Nt_alloc + 1, p.N_sims, N_vertices));
    auto events_from = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, p.N_sims, N_connections));
    auto events_to = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, p.N_sims, N_connections));
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
