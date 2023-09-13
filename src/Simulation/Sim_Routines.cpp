#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Simulation/Sim_Write.hpp>
#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <filesystem>
#include <chrono>

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

void write_allocated_steps(sycl::queue& q, const Sim_Param& p, Sim_Buffers& b, size_t N_steps, std::vector<sycl::event>& dep_events)
{
    N_steps = (N_steps == 0)  ? p.Nt_alloc : N_steps;
    std::chrono::high_resolution_clock::time_point t1, t2;
    N_steps = std::min<size_t>({N_steps, p.Nt_alloc});
    t1 = std::chrono::high_resolution_clock::now();
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Accumulate community state: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    auto state_gs = read_graphseries(q, b.community_state, p, p.Nt_alloc + 1, p.N_communities, acc_event);
    auto event_to_gs = get_N_timesteps(std::forward<const Graphseries_t<uint32_t>>(read_graphseries(q, b.events_to, p, p.Nt_alloc, b.events_to.get_range()[2], dep_events)), N_steps, 0);
    auto event_from_gs = get_N_timesteps(std::forward<const Graphseries_t<uint32_t>>(read_graphseries(q, b.events_from, p, p.Nt_alloc, b.events_from.get_range()[2], dep_events)), N_steps, 0);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Read graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    auto state_gs_write = get_N_timesteps(std::forward<const Graphseries_t<State_t>>(state_gs), N_steps, 1);
    auto state_gs_inf = get_N_timesteps(std::forward<const Graphseries_t<State_t>>(state_gs), N_steps+1, 0);
    write_graphseries(std::forward<const decltype(state_gs_write)>(state_gs_write), p.output_dir, "community_trajectory", true);
    auto connection_events = zip_merge_graphseries(std::forward<const decltype(event_from_gs)>(event_from_gs), std::forward<const decltype(event_to_gs)>(event_to_gs));
    auto inf_gs = sample_infections(std::forward<const Graphseries_t<State_t>>(state_gs_inf), std::forward<const Graphseries_t<uint32_t>>(event_from_gs), std::forward<const Graphseries_t<uint32_t>>(event_to_gs), b.ccm, b.ccm_weights, p.seed, p.compute_range[0]);
    write_graphseries(std::forward<const decltype(inf_gs)>(inf_gs), p.output_dir, "connection_infections", true);
    write_graphseries(std::forward<const decltype(connection_events)>(connection_events), p.output_dir, "connection_events", true);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Inf sample/ write graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
}


void write_initial_steps(sycl::queue& q, const Sim_Param& p, Sim_Buffers& b, std::vector<sycl::event>& dep_events)
{
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range);
    auto state_gs = get_N_timesteps(std::forward<const Graphseries_t<State_t>>(read_graphseries(q, b.community_state, p, p.Nt_alloc + 1, p.N_communities, acc_event)), 1, 0);
    write_graphseries(std::forward<const decltype(state_gs)>(state_gs), p.output_dir, "community_trajectory", true);
}


void run(sycl::queue &q, Sim_Param p, Sim_Buffers &b)
{
    uint32_t N_connections = b.events_from.get_range()[2];
    Sim_Data d(p.Nt_alloc, p.N_sims, p.N_communities, N_connections);
    if ((p.global_mem_size == 0) || p.local_mem_size == 0)
    {
        auto device = q.get_device();
        p.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        p.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    }


    std::vector<sycl::event> events(1);
    q.wait();
    events[0] = initialize_vertices(q, p, b.vertex_state, b.rngs);
    std::filesystem::remove_all(p.output_dir);
    std::filesystem::create_directories(p.output_dir);
    //create directories for all p.N_graphs
    for(int i = 0; i < p.N_graphs; i++)
    {
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/");
    }

    write_initial_steps(q, p, b, events);

    //remove all files in directory
    // write_initial_state(q, p, b, events);
    uint32_t t = 0;
    //make directory
    for (t = 0; t < p.Nt; t++)
    {
        bool is_initial_write = (t == 0);
        if (is_allocated_space_full(t, p.Nt_alloc))
        {
            q.wait();
            write_allocated_steps(q, p, b, p.Nt_alloc, events);
            events[0] = clear_buffer<uint32_t, 3>(q, b.events_from, events);
            events[0] = clear_buffer<uint32_t, 3>(q, b.events_to, events);
            events[0] = move_buffer_row(q, b.vertex_state, p.Nt_alloc, events);
        }
        events = recover(q, p, b.vertex_state, b.rngs, t, events);
        events = infect(q, p, b, t, events);
    }
    write_allocated_steps(q, p, b, t % p.Nt_alloc, events);
    ccms_to_file(b.ccm, p.output_dir);

    auto p_Is = get_N_timesteps(std::forward<const Graphseries_t<float>>(read_graphseries(q, b.p_Is, p, p.Nt, b.p_Is.get_range()[2], events)), p.Nt, 0);
    write_graphseries(std::forward<const Graphseries_t<float>>(p_Is), p.output_dir, "p_I", false);

    p.dump(p.output_dir + "/Sim_Param.json");

}



void run(sycl::queue& q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_list, const std::vector<std::vector<uint32_t>>& vcm, const std::vector<std::vector<uint32_t>>& ecm)
{

    auto b = Sim_Buffers::make(q, p, edge_list, vcm, ecm, {});
    b.validate_sizes(p);
    run(q, p, b);
    for(int graph_idx = 0; graph_idx < edge_list.size(); graph_idx++)
    {
        write_edgelist(p.output_dir + "/Graph_" + std::to_string(graph_idx) + "/edgelist.csv", edge_list[graph_idx]);
        write_vector(p.output_dir + "/Graph_" + std::to_string(graph_idx) + "/vcm.csv", vcm[graph_idx]);
        write_vector(p.output_dir + "/Graph_" + std::to_string(graph_idx) + "/ecm.csv", ecm[graph_idx]);
    }
}

auto matrix_linearize(const std::vector<std::vector<float>>& vecs)
{
    std::vector<float> out;
    out.reserve(vecs.size() * vecs[0].size());
    for(auto&& v: vecs)
    {
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

auto dataframe_linearize(const std::vector<std::vector<std::vector<float>>>& df)
{
    std::vector<float> result(df.size() * df[0].size() * df[0][0].size());
    for(int i = 0; i < df.size(); i++)
    {
        for(int j = 0; j < df[0].size(); j++)
        {
            for(int k = 0; k < df[0][0].size(); k++)
            {
                result[i * df[0].size() * df[0][0].size() + j * df[0][0].size() + k] = df[i][j][k];
            }
        }
    }
    return result;
}

void p_I_run(sycl::queue& q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_list, const std::vector<std::vector<uint32_t>>& vcm, const std::vector<std::vector<uint32_t>>& ecm, const std::vector<std::vector<std::vector<float>>>& p_Is)
{
    auto p_I_lin = dataframe_linearize(p_Is);
    auto b = Sim_Buffers::make(q, p, edge_list, ecm, vcm, p_I_lin);
    run(q, p, b);
}
