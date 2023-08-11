
#include <Sycl_Graph/Simulation.hpp>

#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/SIR_Dynamics.hpp>
#include <Sycl_Graph/SIR_Infection_Sampling.hpp>
#include <Sycl_Graph/Profiling.hpp>
#include <algorithm>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>




sycl::event Simulator::accumulate_community_state(std::vector<sycl::event> dep_events)
{

    auto range = sycl::nd_range(sycl::range<1>(N_sims), sycl::range<2>(N_communities, Nt+1));
    auto N_vertices = this->N_vertices;
    // auto N_per_thread = static_cast<uint32_t>(std::ceil(static_cast<float>(N_communities*(Nt+1)) / static_cast<float>(N_wg)));
    return q.submit([&](sycl::handler& h)
    {
        auto v_acc = trajectory.template get_access<sycl::access::mode::read>(h);
        auto vcm_acc = vcm.template get_access<sycl::access::mode::read>(h);
        auto community_state_acc = community_state.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(range, [=](sycl::nd_item<2> it)
        {
            auto gid = it.get_global_id()[0];
            auto lid = it.get_local_id();
            auto c_idx = lid[0];
            auto t = lid[1];
            community_state_acc[gid][t][c_idx] = {0,0,0};
            for(int i = 0; i < N_vertices;i++)
            {
                SIR_State state = v_acc[gid][t][vcm_acc[i]];
                community_state_acc[gid][t][c_idx][state]++;
            }
        });
    });
}


std::vector<sycl::event> enqueue_timeseries(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, Individual_Buffers &ib, std::vector<sycl::event>& dep_events)
{
    std::chrono::steady_clock::time_point begin, end;
    std::cout << "Enqueueing timeseries..." << std::endl;
    std::vector<sycl::event> events = dep_events;
    events.reserve(4*p.Nt);
    for (int t = 0; t < p.Nt; t++)
    {

        std::cout << "Enqueueing timestep " << t << " of " << p.Nt << "..." << std::endl;
        std::cout << "Recovery [";
        begin = std::chrono::steady_clock::now();
        events.push_back(recover(q, t, events, p.p_R, ib.seeds, ib.trajectory));
        end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms]" << std::endl;
        std::cout << "Infection [";
        begin = std::chrono::steady_clock::now();
        auto inf_events = infect(q, cb.ecm, ib.p_Is, ib.seeds, ib.events_from, ib.events_to, ib.trajectory, cb.edge_from, cb.edge_to, ib.infections_from, ib.infections_to, t, cb.N_connections(), events[3*t+1]);
        //append to events
        events.insert(events.end(), inf_events.begin(), inf_events.end());
        end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms]" << std::endl;
    }
    return events;
}



void write_to_file(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, Individual_Buffers &ib, const std::vector<uint32_t> &vcm, const std::string &output_dir, std::vector<sycl::event>& events, uint32_t sim_idx, uint32_t seed)
{
    std::ofstream kernel_profile_f(output_dir + "/kernel_profile.log");
    auto write_profiling_timestep = [&](auto& e0, auto& e1, auto& e2, uint32_t t){kernel_profile_f << "Timestep " << t << ": ";
    kernel_profile_f << "Recovery: " << get_event_execution_time(e0) << ", ";
    kernel_profile_f << "Infection: " << get_event_execution_time(e1) << std::endl;
    kernel_profile_f << "Accumulation: " << get_event_execution_time(e2) << std::endl;
    };
    std::for_each(events.begin(), events.end(), [&](auto& e){e.wait();});
    kernel_profile_f << "Allocation time: " << get_event_execution_time(events[0]) << std::endl;
    kernel_profile_f << "Infection Dynamics kernel execution time:" << std::endl;
    for(int i = 0; i < p.Nt; i++)
    {
        write_profiling_timestep(events[3*i+1], events[3*i+2], events[3*i+3], i);
    }
    std::vector<sycl::event> read_events(3);
    auto buffer_data = ib.read_buffers(q, events);
    auto community_state = to_community_state(q, buffer_data.vertex_state, vcm);

    auto connection_infections = sample_infections(community_state, buffer_data.from_events, buffer_data.to_events, cb.ccm, cb.ccm_weights, seed, p.max_infection_samples);

    auto connection_events = column_zip_2D(buffer_data.from_events, buffer_data.to_events);


    std::filesystem::create_directories(output_dir);

    std::ofstream community_traj_f(output_dir + "community_trajectory_" +
                                   std::to_string(sim_idx) + ".csv");
    std::ofstream connection_events_f(output_dir + "connection_events_" +
                                      std::to_string(sim_idx) + ".csv");

    std::ofstream connection_infections_f(output_dir + "connection_infections_" +
                                          std::to_string(sim_idx) + ".csv");

    std::for_each(community_state.begin(), community_state.end(),
                  [&](auto &community_trajectory_i)
                  {
                      linewrite(community_traj_f, community_trajectory_i);
                  });
    std::for_each(connection_events.begin(),
                  connection_events.end(),
                  [&](auto &connection_events_i)
                  {
                      linewrite(connection_events_f, connection_events_i);
                  });

    std::for_each(connection_infections.begin(),
                  connection_infections.end(),
                  [&](auto &connection_infections_i)
                  {
                      linewrite(connection_infections_f, connection_infections_i);
                  });
}

std::vector<sycl::event> single_enqueue(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, Individual_Buffers &ib, const std::string output_dir)
{
    std::vector<sycl::event> dep_events = cb.events;
    dep_events.push_back(ib.initialize_trajectory(q, p.p_I0, p.p_R0, ib.events));
    return enqueue_timeseries(q, p, cb, ib, dep_events);
}
uint32_t max_work_group_size(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    return max_wg_size;
}

void simulate(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<std::vector<float>> &p_Is, const std::string output_dir, uint32_t N_simulations, uint32_t seed)
{
    auto N_edges = edge_list.size();
    auto N_wg = max_work_group_size(q);
    std::vector<std::vector<sycl::event>> events(N_simulations, std::vector<sycl::event>(3*p.Nt+1));

    auto seeds = generate_seeds(N_simulations, seed);
    std::vector<Individual_Buffers> ibs;
    ibs.reserve(N_simulations);

    std::generate_n(std::back_inserter(ibs), N_simulations, [&, n = 0]() mutable
                    { return Individual_Buffers(q, p_Is, cb.N_connections(), p.Nt, p.N_communities, p.N_pop, N_edges, N_wg, seed); });
    std::transform(ibs.begin(), ibs.end(), events.begin(), [&](auto &ib)
                   { return single_enqueue(q, p, cb, ib, output_dir); });

    for (int i = 0; i < N_simulations; i++)
    {
        write_to_file(q, p, cb, ibs[i], vcm, output_dir, events[i], i, seeds[i]);
    }
}

void excite_simulate(sycl::queue &q, const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir, uint32_t N_simulations, uint32_t seed)
{
    std::chrono::steady_clock::time_point begin, end;

    begin = std::chrono::steady_clock::now();
    auto N_edges = edge_list.size();
    auto N_wg = max_work_group_size(q);
    auto seeds = generate_seeds(N_simulations, seed);
    std::vector<std::vector<sycl::event>> events(N_simulations, std::vector<sycl::event>(3*p.Nt+1));
    auto cb = allocate_common_buffers(q, edge_list, vcm, p.Nt, p.N_communities, seed);
    uint32_t N_connections = cb.N_connections();
    auto p_Is_vec = generate_floats(N_simulations, p.Nt, N_connections, p_I_min, p_I_max, seed);
    auto ibs = Individual_Buffers::make(q, p_Is_vec, N_connections, p.Nt, p.N_communities, p.N_pop, N_edges, N_wg, seeds);
    end = std::chrono::steady_clock::now();
    std::cout << "Buffer construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    begin = std::chrono::steady_clock::now();
    std::transform(ibs.begin(), ibs.end(), events.begin(), [&](auto &ib)
                   { return single_enqueue(q, p, cb, ib, output_dir); });
    end = std::chrono::steady_clock::now();
    std::cout << "Enqueue: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    std::cout << "Reading to files..." << std::endl;
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N_simulations; i++)
    {
        write_to_file(q, p, cb, ibs[i], vcm, output_dir, events[i], i, seeds[i]);
    }
    end = std::chrono::steady_clock::now();
}
