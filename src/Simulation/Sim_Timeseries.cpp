#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Simulation/Sim_Write.hpp>
void community_state_to_timeseries(sycl::queue &q,
                                   sycl::buffer<SIR_State, 3> &vertex_state,
                                   sycl::buffer<State_t, 3> &community_state,
                                   std::vector<std::vector<std::vector<State_t>>> &community_timeseries,
                                   sycl::buffer<uint32_t> &vcm,
                                   uint32_t t_offset,
                                   sycl::range<1> compute_range,
                                   sycl::range<1> wg_range,
                                   std::vector<sycl::event> &dep_events)
{

    uint32_t N_sims = vertex_state.get_range()[1];
    uint32_t Nt_alloc = vertex_state.get_range()[0] - 1;
    uint32_t N_communities = community_state.get_range()[2];
    std::vector<sycl::event> acc_event(1);
    sycl::event event;
    std::vector<State_t> community_state_flat((Nt_alloc+1) * N_sims * N_communities);
    acc_event[0] = accumulate_community_state(q, dep_events, vertex_state, vcm, community_state, Nt_alloc + 1, compute_range, wg_range);
    event = read_buffer<State_t, 3>(community_state, q, community_state_flat, acc_event);
    event.wait();
    auto cs = vector_remap(community_state_flat, Nt_alloc+1, N_sims, N_communities);
    auto Nt_max = community_timeseries[0].size();
    for (size_t i0 = 0; i0 < N_sims; i0++)
    {
        for (size_t i2 = 0; i2 < (Nt_alloc+1); i2++)
        {
            if ((t_offset + i2) >= Nt_max)
                break;
            for (size_t i1 = 0; i1 < N_communities; i1++)
            {
                auto state = cs[i2][i0][i1];
                community_timeseries[i0][t_offset + i2][i1] = state;
            }
        }
    }
}

void community_state_append_to_file(sycl::queue &q,
                                   sycl::buffer<SIR_State, 3> &vertex_state,
                                   sycl::buffer<State_t, 3> &community_state,
                                   sycl::buffer<uint32_t> &vcm,
                                   sycl::range<1> compute_range,
                                   sycl::range<1> wg_range,
                                   const std::string& output_dir,
                                   std::vector<sycl::event> &dep_events)
{
    uint32_t N_sims = vertex_state.get_range()[1];
    uint32_t Nt_alloc = vertex_state.get_range()[0] - 1;
    uint32_t N_communities = community_state.get_range()[2];
    std::vector<sycl::event> acc_event(1);
    sycl::event event;
    std::vector<State_t> community_state_flat((Nt_alloc+1) * N_sims * N_communities);
    acc_event[0] = accumulate_community_state(q, dep_events, vertex_state, vcm, community_state, Nt_alloc + 1, compute_range, wg_range);
    event = read_buffer<State_t, 3>(community_state, q, community_state_flat, acc_event);
    event.wait();
    auto cs = vector_remap(community_state_flat, Nt_alloc+1, N_sims, N_communities);
    std::vector<std::vector<std::vector<State_t>>> community_timeseries(N_sims, std::vector<std::vector<State_t>>(Nt_alloc+1, std::vector<State_t>(N_communities)));
    auto Nt_max = community_timeseries[0].size();
    for (size_t i0 = 0; i0 < N_sims; i0++)
    {
        for (size_t i2 = 0; i2 < (Nt_alloc+1); i2++)
        {
            if ((i2) >= Nt_max)
                break;
            for (size_t i1 = 0; i1 < N_communities; i1++)
            {
                auto state = cs[i2][i0][i1];
                community_timeseries[i0][i2][i1] = state;
            }
        }
    }
    timeseries_to_file(community_timeseries, output_dir + "/community_trajectory", true);
}
void community_state_init_to_file(sycl::queue &q,
                                   sycl::buffer<SIR_State, 3> &vertex_state,
                                   sycl::buffer<State_t, 3> &community_state,
                                   sycl::buffer<uint32_t> &vcm,
                                   sycl::range<1> compute_range,
                                   sycl::range<1> wg_range,
                                   const std::string& output_dir,
                                   std::vector<sycl::event> &dep_events)
{
    uint32_t N_sims = vertex_state.get_range()[1];
    uint32_t Nt_alloc = vertex_state.get_range()[0] - 1;
    uint32_t N_communities = community_state.get_range()[2];
    std::vector<sycl::event> acc_event(1);
    sycl::event event;
    std::vector<State_t> community_state_flat((Nt_alloc+1) * N_sims * N_communities);
    acc_event[0] = accumulate_community_state(q, dep_events, vertex_state, vcm, community_state, Nt_alloc + 1, compute_range, wg_range);
    event = read_buffer<State_t, 3>(community_state, q, community_state_flat, acc_event);
    event.wait();
    auto cs = vector_remap(community_state_flat, Nt_alloc+1, N_sims, N_communities);
    std::vector<std::vector<std::vector<State_t>>> community_timeseries(N_sims, std::vector<std::vector<State_t>>(1, std::vector<State_t>(N_communities)));
    auto Nt_max = community_timeseries[0].size();
    for (size_t i0 = 0; i0 < N_sims; i0++)
    {
        for (size_t i2 = 0; i2 < (1); i2++)
        {
            if ((i2) >= Nt_max)
                break;
            for (size_t i1 = 0; i1 < N_communities; i1++)
            {
                auto state = cs[i2][i0][i1];
                community_timeseries[i0][i2][i1] = state;
            }
        }
    }
    timeseries_to_file(community_timeseries, output_dir + "/community_trajectory");
}

void connection_events_to_timeseries(sycl::queue &q,
                                     sycl::buffer<uint32_t, 3> events_from,
                                     sycl::buffer<uint32_t, 3> &events_to,
                                     std::vector<std::vector<std::vector<uint32_t>>> &events_from_timeseries,
                                     std::vector<std::vector<std::vector<uint32_t>>> &events_to_timeseries,
                                     sycl::buffer<uint32_t> &vcm,
                                     uint32_t t_offset,
                                     std::vector<sycl::event> &dep_events)
{

    uint32_t Nt_alloc = events_from.get_range()[0];
    uint32_t N_sims = events_from.get_range()[1];
    uint32_t N_connections = events_from.get_range()[2];
    uint32_t Nt_max = events_from_timeseries[0].size();
    auto read_remap_to_ts = [&](auto &buf)
    {
        std::vector<uint32_t> event_buf_flat(Nt_alloc * N_sims * N_connections);
        auto evt = read_buffer<uint32_t, 3>(buf, q, event_buf_flat, dep_events);
        evt.wait();
        return vector_remap(event_buf_flat, Nt_alloc, N_sims, N_connections);
    };
    auto e_from = read_remap_to_ts(events_from);
    auto e_to = read_remap_to_ts(events_to);

    for (size_t i0 = 0; i0 < N_sims; i0++)
    {
        for (size_t i2 = 0; i2 < Nt_alloc; i2++)
        {
            if ((t_offset + i2) >= Nt_max)
                break;
                for (size_t i1 = 0; i1 < N_connections; i1++)
                {
                    auto from_state = e_from[i2][i0][i1];
                    auto to_state = e_to[i2][i0][i1];
                    events_from_timeseries[i0][t_offset + i2][i1] = from_state;
                    events_to_timeseries[i0][t_offset + i2][i1] = to_state;
                }
        }
    }
}

void connection_events_append_to_file(sycl::queue &q,
                                     sycl::buffer<uint32_t, 3> events_from,
                                     sycl::buffer<uint32_t, 3> &events_to,
                                     sycl::buffer<uint32_t> &vcm,
                                     const std::string& output_dir,
                                     std::vector<sycl::event> &dep_events,
                                     bool append)
{


    uint32_t Nt_alloc = events_from.get_range()[0];
    uint32_t N_sims = events_from.get_range()[1];
    uint32_t N_connections = events_from.get_range()[2];
    std::vector<std::vector<std::vector<uint32_t>>> events_to_timeseries(N_sims, std::vector<std::vector<uint32_t>>(Nt_alloc, std::vector<uint32_t>(N_connections)));
    std::vector<std::vector<std::vector<uint32_t>>> events_from_timeseries(N_sims, std::vector<std::vector<uint32_t>>(Nt_alloc, std::vector<uint32_t>(N_connections)));
    uint32_t Nt_max = events_from_timeseries[0].size();
    auto read_remap_to_ts = [&](auto &buf)
    {
        std::vector<uint32_t> event_buf_flat(Nt_alloc * N_sims * N_connections);
        auto evt = read_buffer<uint32_t, 3>(buf, q, event_buf_flat, dep_events);
        evt.wait();
        return vector_remap(event_buf_flat, Nt_alloc, N_sims, N_connections);
    };
    auto e_from = read_remap_to_ts(events_from);
    auto e_to = read_remap_to_ts(events_to);

    for (size_t i0 = 0; i0 < N_sims; i0++)
    {
        for (size_t i2 = 0; i2 < Nt_alloc; i2++)
        {
            if ((i2) >= Nt_max)
                break;
                for (size_t i1 = 0; i1 < N_connections; i1++)
                {
                    auto from_state = e_from[i2][i0][i1];
                    auto to_state = e_to[i2][i0][i1];
                    events_from_timeseries[i0][i2][i1] = from_state;
                    events_to_timeseries[i0][i2][i1] = to_state;
                }
        }
    }
    events_to_file(events_from_timeseries, events_to_timeseries, output_dir + "/connection_events", append);
}
