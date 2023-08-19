#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <Sycl_Graph/Simulation/Sim_Write.hpp>
#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Simulation/Sim_Data.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>
void community_state_to_timeseries(sycl::queue &q,
                                   sycl::buffer<SIR_State, 3> &vertex_state,
                                   sycl::buffer<State_t, 3> &community_state,
                                   std::vector<std::vector<std::vector<State_t>>>& community_timeseries,
                                   sycl::buffer<uint32_t> &vcm,
                                   uint32_t t_offset,
                                   std::vector<sycl::event> &dep_events)
{
    uint32_t N_sims = vertex_state.get_range()[1];
    uint32_t Nt_alloc = vertex_state.get_range()[0]-1;
    uint32_t N_communities = community_state.get_range()[2];
    auto [compute_range, wg_range] = sim_ranges(q, N_sims);
    std::vector<sycl::event> acc_event(1);
    sycl::event event;
    std::vector<State_t> community_state_flat(Nt_alloc*N_sims*N_communities);
    acc_event[0] = accumulate_community_state(q, dep_events, vertex_state, vcm, community_state, Nt_alloc, compute_range, wg_range);
    event = read_buffer<State_t, 3>(community_state, q, community_state_flat, acc_event);
    event.wait();
    auto cs = vector_remap(community_state_flat, Nt_alloc, N_sims, N_communities);
    for (size_t i0 = 0; i0 < N_sims; i0++)
    {
        for (size_t i2 = 0; i2 < Nt_alloc; i2++)
        {
            auto i2_corrected = std::max<int>((int)(i2 + t_offset + 1) - Nt_alloc, 0);
                for (size_t i1 = 0; i1 < N_communities; i1++)
                {
                    auto state = cs[i2][i0][i1];
                    community_timeseries[i0][i2_corrected][i1] = state;
                }
        }
    }
}

void connection_events_to_timeseries(sycl::queue &q,
                                   sycl::buffer<uint32_t, 3> events_from,
                                   sycl::buffer<uint32_t, 3>& events_to,
                                   std::vector<std::vector<std::vector<uint32_t>>>& events_from_timeseries,
                                   std::vector<std::vector<std::vector<uint32_t>>>& events_to_timeseries,
                                   sycl::buffer<uint32_t> &vcm,
                                   uint32_t t_offset,
                                   std::vector<sycl::event> &dep_events)
{
    auto [compute_range, wg_range] = get_work_group_ranges();

    uint32_t Nt_alloc = events_from.get_range()[0];
    uint32_t N_sims = events_from.get_range()[1];
    uint32_t N_connections = events_from.get_range()[2];
    auto read_remap_to_ts = [&](auto& buf)
    {
        std::vector<uint32_t> event_buf_flat(Nt_alloc*N_sims*N_connections);
        auto evt= = read_buffer<uint32_t, 3>(buf, q, event_buf_flat, dep_events);
        evt.wait();
        return vector_remap(event_buf_flat, Nt_alloc, N_sims, N_connections);
    };

    auto e_from = read_remap_to_ts(events_from);
    auto e_to = read_remap_to_ts(events_to);

    for (size_t i0 = 0; i0 < N_sims; i0++)
    {
        for (size_t i2 = 0; i2 < Nt_alloc; i2++)
        {
            auto i2_corrected = std::max<int>((int)(i2 + t_offset + 1) - Nt_alloc, 0);
            if (i2_corrected < Nt)
                for (size_t i1 = 0; i1 < N_connections(); i1++)
                {
                    auto from_state = e_from[i2][i0][i1];
                    auto to_state = e_to[i2][i0][i1];
                    events_from_timeseries[i0][i2_corrected][i1] = from_state;
                    events_to_timeseries[i0][i2_corrected][i1] = to_state;
                }
        }
    }
}

bool is_allocated_space_full(uint32_t t, uint32_t Nt_alloc)
{
    return ((t != 0) && (t % Nt_alloc == 0));
}

void run(sycl::queue& q, const Sim_Param& p, Sim_Buffers& b)
{
    Sim_Data d(p.Nt, p.N_sims, p.N_communities, p.N_connections);
    std::vector<sycl::event> events(1);
    events[0] = initialize_vertices(q, p, b.vertex_state, b.rngs);

    for(int t = 0; t < p.Nt; t++)
    {
        events = recover(q, p, b.vertex_state, b.rngs, t % p.Nt_alloc, events);
        events = infect(q, p, b, events);
        if (is_allocated_space_full(t, p.Nt_alloc))
        {
            community_state_to_timeseries(q, b.vertex_state, b.community_state, d.community_timeseries, b.vcm, t, events);
            connection_events_to_timeseries(q, b.events_from, b.events_to, d.events_from_timeseries, d.events_to_timeseries, b.vcm, t, events);
        }
    }

    d.connection_infections = sample_from_connection_events(d.state_timeseries, d.events_from_timeseries, d.events_to_timeseries);

    timeseries_to_file(b.state_timeseries, p.output_dir + "/community_trajectory");
    events_to_file(d.events_from_timeseries, d.events_to_timeseries, p.output_dir + "/connection_events");
    timeseries_to_file(d.connection_infections, p.output_dir + "/connection_infections");

}
