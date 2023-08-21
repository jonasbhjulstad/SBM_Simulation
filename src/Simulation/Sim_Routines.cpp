#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Simulation/Sim_Write.hpp>
#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>

sycl::event move_buffer_row(sycl::queue& q, sycl::buffer<SIR_State,3>& buf, uint32_t row, std::vector<sycl::event>& dep_events)
{
    auto Nt = buf.get_range()[0];
    auto N_sims = buf.get_range()[1];
    auto N_vertices = buf.get_range()[2];
    return q.submit([&](sycl::handler &h)
                                {
            auto start_acc = sycl::accessor<SIR_State, 3, sycl::access_mode::write>(buf, h, sycl::range<3>(1,N_sims, N_vertices), sycl::range<3>(0,0,0));
            auto end_acc = sycl::accessor<SIR_State, 3, sycl::access_mode::read>(buf, h, sycl::range<3>(1,N_sims, N_vertices), sycl::range<3>(row,0,0));
            h.copy(end_acc, start_acc); });
}

bool is_allocated_space_full(uint32_t t, uint32_t Nt_alloc)
{
    return ((t != 0) && (t % (Nt_alloc) == 0));
}

void run_allocated(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b)
{
    uint32_t N_connections = b.events_from.get_range()[2];
    Sim_Data d(p.Nt, p.N_sims, p.N_communities, N_connections);
    std::vector<sycl::event> events(1);
    events[0] = initialize_vertices(q, p, b.vertex_state, b.rngs);
    community_state_to_timeseries(q, b.vertex_state, b.community_state, d.state_timeseries, b.vcm, 0, p.compute_range, p.wg_range, events);
    q.wait();
    uint32_t t = 0;
    for (t = 0; t < p.Nt; t++)
    {
        if (is_allocated_space_full(t, p.Nt_alloc))
        {

            print_timestep(q, events, b.events_from, b.events_to, b.vertex_state, b.vcm, p);
            community_state_to_timeseries(q, b.vertex_state, b.community_state, d.state_timeseries, b.vcm, (t - p.Nt_alloc), p.compute_range, p.wg_range, events);
            connection_events_to_timeseries(q, b.events_from, b.events_to, d.events_from_timeseries, d.events_to_timeseries, b.vcm, (t - p.Nt_alloc), events);
            events[0] = move_buffer_row(q, b.vertex_state, p.Nt_alloc, events);
            q.wait();
        }
        events = recover(q, p, b.vertex_state, b.rngs, t, events);
        q.wait();

        events = infect(q, p, b, t, events);
        q.wait();
    }
    auto last_offset = p.Nt - (t % p.Nt_alloc);
    community_state_to_timeseries(q, b.vertex_state, b.community_state, d.state_timeseries, b.vcm, last_offset, p.compute_range, p.wg_range, events);
    connection_events_to_timeseries(q, b.events_from, b.events_to, d.events_from_timeseries, d.events_to_timeseries, b.vcm, last_offset, events);

    timeseries_to_file(d.state_timeseries, p.output_dir + "/community_trajectory");
    events_to_file(d.events_from_timeseries, d.events_to_timeseries, p.output_dir + "/connection_events");
    d.connection_infections = sample_from_connection_events(d.state_timeseries, d.events_from_timeseries, d.events_to_timeseries, b.ccm, b.ccm_weights, p.seed);

    timeseries_to_file(d.connection_infections, p.output_dir + "/connection_infections");
}


void run(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b)
{
    uint32_t N_connections = b.events_from.get_range()[2];
    Sim_Data d(p.Nt, p.N_sims, p.N_communities, N_connections);
    std::vector<sycl::event> events(1);
    events[0] = initialize_vertices(q, p, b.vertex_state, b.rngs);
    community_state_init_to_file(q, b.vertex_state, b.community_state, b.vcm, p.compute_range, p.wg_range, p.output_dir, events);
    q.wait();
    uint32_t t = 0;
    for (t = 0; t < p.Nt; t++)
    {
        bool is_initial_write = (t == 0);
        if (is_allocated_space_full(t, p.Nt_alloc))
        {
            print_timestep(q, events, b.events_from, b.events_to, b.vertex_state, b.vcm, p);
            events[0] = move_buffer_row(q, b.vertex_state, p.Nt_alloc, events);
            community_state_append_to_file(q, b.vertex_state, b.community_state, b.vcm, p.compute_range, p.wg_range, p.output_dir, events);
            connection_events_append_to_file(q, b.events_from, b.events_to, b.vcm, p.output_dir, events, is_initial_write);
            d.connection_infections = sample_from_connection_events(d.state_timeseries, d.events_from_timeseries, d.events_to_timeseries, b.ccm, b.ccm_weights, p.seed);
            timeseries_to_file(d.connection_infections, p.output_dir + "/connection_infections", is_initial_write);
            q.wait();
        }
        events = recover(q, p, b.vertex_state, b.rngs, t, events);
        q.wait();

        events = infect(q, p, b, t, events);
        q.wait();
    }
    auto last_offset = p.Nt - (t % p.Nt_alloc);
}
