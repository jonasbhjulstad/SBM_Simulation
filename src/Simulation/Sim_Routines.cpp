#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Simulation/Sim_Write.hpp>
#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <filesystem>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>

sycl::event reset_state_index(sycl::queue& q, Sim_Param& p, Sim_Buffers& b, std::vector<sycl::event>& dep_events)
{
    auto N_vertices = p.N_pop * p.N_communities;
    return q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_events);
        h.parallel_for(p.N_sims, [=](sycl::item<1> it)
        {
            auto offset_0 = get_linear_offset(p.N_sims, p.Nt_alloc, N_vertices, it[0], 0, 0);
            auto offset_end = get_linear_offset(p.N_sims, p.Nt_alloc, N_vertices, it[0], p.Nt_alloc, 0);
            for(auto i = offset_0; i < offset_end; i++)
            {
                b.vertex_state[offset_0 + i] = b.vertex_state[offset_end + i];
            }
        });
    });
}

bool is_allocated_space_full(uint32_t t, uint32_t Nt_alloc)
{
    return ((t != 0) && (t % (Nt_alloc) == 0));
}

void read_allocated_space_to_file(sycl::queue& q, const Sim_Param& p, Sim_Buffers& b, std::vector<sycl::event>& events, const std::string& output_dir, uint32_t Nt, bool append)
{
    auto state_timeseries = read_community_state(q, p, b.community_state, events);
    auto [events_from, events_to] = read_connection_events(q, p, b, events);
    auto events_from_timeseries = events_from;
    auto events_to_timeseries = events_to;
    auto connection_infections = sample_from_connection_events(state_timeseries, events_from_timeseries, events_to_timeseries, b.ccm, b.ccm_weights, p.seed, p.compute_range[0]);
    auto merged_events = zip_merge_timeseries(get_N_timesteps(events_from_timeseries, Nt), get_N_timesteps(events_to_timeseries, Nt));

    write_timeseries(get_N_timesteps(state_timeseries, Nt), output_dir, "community_trajectory", append);
    write_timeseries(merged_events, output_dir, "connection_events", append);
    write_timeseries(get_N_timesteps(connection_infections, Nt), output_dir, "connection_infections", append);
}



void run(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, const std::string& output_dir)
{
    uint32_t N_connections = b.N_connections;
    Sim_Data d(p.N_graphs, p.N_sims, p.Nt_alloc, p.N_communities, N_connections);
    std::vector<sycl::event> events(1);
    q.wait();
    events[0] = initialize_vertices(q, p, b);
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);

    q.wait();
    uint32_t t = 0;
    //make directory
    for (t = 0; t < p.Nt; t++)
    {
        bool is_initial_write = (t == 0);
        if (is_allocated_space_full(t, p.Nt_alloc) && false)
        {
            read_allocated_space_to_file(q, p, b, events, output_dir, p.Nt_alloc, !is_initial_write);
            events[0] = reset_state_index(q, p, b, events);
            q.wait();
        }
        events = recover(q, p, b, t, events);
        q.wait();
        events = infect(q, p, b, t, events);
        q.wait();
    }
    auto last_offset = p.Nt - (t % p.Nt_alloc);
    read_allocated_space_to_file(q, p, b, events, output_dir, last_offset, true);
}
