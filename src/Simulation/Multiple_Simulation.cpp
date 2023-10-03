#include <Sycl_Graph/Simulation/Multiple_Simulation.hpp>
#include <Sycl_Graph/Simulation/State_Accumulation.hpp>
void reset_directories(const Multiple_Sim_Param_t &p)
{
    for (auto &&po : p.p_out)
    {
        auto po_str = float_to_decimal_string(po, 2);
        auto graph_dir = p.output_dir + "/" + po_str + "/Graph_";
        for (int i = 0; i < p.N_graphs; i++)
        {
            std::filesystem::remove_all(graph_dir + std::to_string(i) + "/");
        }
        std::filesystem::create_directories(p.output_dir);
        // create directories for all p.N_graphs
        for (int i = 0; i < p.N_graphs; i++)
        {
            std::filesystem::create_directories(graph_dir + std::to_string(i) + "/Trajectories/");
            std::filesystem::create_directories(graph_dir + std::to_string(i) + "/p_Is/");
            std::filesystem::create_directories(graph_dir + std::to_string(i) + "/Connection_Infections/");
            std::filesystem::create_directories(graph_dir + std::to_string(i) + "/Connection_Events/");
        }
    }
}

void write_allocated_steps(sycl::queue &q, const Multiple_Sim_Param_t &p, Sim_Buffers &b, size_t N_steps, std::vector<sycl::event> &dep_events)
{
    N_steps = (N_steps == 0) ? p.Nt_alloc : N_steps;
    std::chrono::high_resolution_clock::time_point t1, t2;
    N_steps = std::min<size_t>({N_steps, p.Nt_alloc});
    t1 = std::chrono::high_resolution_clock::now();
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range, p.N_sims);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Accumulate community state: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    auto state_df = read_3D_buffer(q, b.community_state, p.N_graphs, {acc_event});
    auto event_df = read_3D_buffer(q, b.accumulated_events, p.N_graphs, {acc_event})({0, 0, 0, 0}, {p.N_graphs, p.N_sims, N_steps, b.N_connections_max});
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Read graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    state_df.resize_dim(2, N_steps + 1);
    auto state_df_write = state_df;
    state_df_write.resize_dim(2, N_steps + 1);
    for (int i = 0; i < p.N_graphs; i++)
    {
        state_df[i].resize_dim(2, b.N_communities_vec[i]);
    }
    auto df_range = state_df.get_ranges();
    auto sim_idx = 0;
    for (int i = 0; i < p.p_out.size(); i++)
    {

        auto p_out = p.p_out[i];
        auto Graph_dir = p.output_dir + "/" + float_to_decimal_string(p_out) + "/Graph_";

        write_dataframe(Graph_dir, "/Trajectories/community_trajectory_", state_df_write.slice(sim_idx, sim_idx + p.N_sims), true, {1, 0});
        write_dataframe(Graph_dir, "/Connection_Events/connection_events_", event_df.slice(sim_idx, sim_idx + p.N_sims), true);

        auto inf_gs = sample_infections(state_df, event_df, b.ccm, p.seed);
        write_dataframe(Graph_dir, "/Connection_Infections/connection_infections_", inf_gs.slice(sim_idx, sim_idx + p.N_sims), true);
        sim_idx += p.N_sims;
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Inf sample/ write graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
}

void write_initial_steps(sycl::queue &q, const Multiple_Sim_Param_t &p, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
{
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range, p.N_sims);
    auto state_df = read_3D_buffer(q, b.community_state, p.N_graphs, {acc_event});
    state_df.resize_dim(2, 1);
    auto sim_idx = 0;
    for (int i = 0; i < p.p_out.size(); i++)
    {

        auto p_out = p.p_out[i];
        auto Graph_dir = p.output_dir + "/" + float_to_decimal_string(p_out) + "/Graph_";
        write_dataframe(Graph_dir, "/Trajectories/community_trajectory_", state_df.slice(sim_idx, sim_idx + p.N_sims), false);
        sim_idx += p.N_sims;
    }

}

void run(sycl::queue &q, Multiple_Sim_Param_t p, Sim_Buffers &b)
{
    if ((p.global_mem_size == 0) || p.local_mem_size == 0)
    {
        auto device = q.get_device();
        p.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        p.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    }

    std::vector<sycl::event> events(1);
    q.wait();
    events[0] = initialize_vertices(q, p.to_sim_param(), b.vertex_state, b.rngs);
    reset_directories(p);
    write_initial_steps(q, p, b, events);

    // remove all files in directory
    //  write_initial_state(q, p, b, events);
    uint32_t t = 0;
    // make directory
    for (t = 0; t < p.Nt; t++)
    {
        bool is_initial_write = (t == 0);
        if (is_allocated_space_full(t, p.Nt_alloc))
        {
            q.wait();
            write_allocated_steps(q, p, b, p.Nt_alloc, events);
            events[0] = clear_buffer<uint32_t, 3>(q, b.accumulated_events, events);
            // events[0] = clear_buffer<uint8_t, 3>(q, b.edge_events, events);
            events[0] = move_buffer_row(q, b.vertex_state, p.Nt_alloc, events);
        }
        events = recover(q, p.to_sim_param(), b.vertex_state, b.rngs, t, events);
        events = infect(q, p.to_sim_param(), b, t, events);
        std::cout << "t: " << t << "\n";
    }
    write_allocated_steps(q, p, b, t % p.Nt_alloc, events);

    if_false_throw(b.ccm.size() == p.N_graphs, "ccm size does not match N_graphs");

    auto p_I_df = read_3D_buffer(q, b.p_Is, p.N_graphs, events);
    auto sim_idx = 0;
    for (int i = 0; i < p.p_out.size(); i++)
    {
        auto p_out = p.p_out[i];
        auto Graph_dir = p.output_dir + "/" + float_to_decimal_string(p_out) + "/Graph_";
        write_dataframe(Graph_dir, "/ccm.csv", b.ccm, false);
        write_dataframe(Graph_dir, "/p_Is/p_I_", p_I_df.slice(sim_idx, sim_idx + p.N_sims), true);
        sim_idx += p.N_sims;
    }

    p.dump(p.output_dir + "/Sim_Param.json");
}

// void write_initial_steps(sycl::queue &q, const Multiple_Sim_Param_t &p, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
// {
//     auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range, p.N_sims);
//     auto state_df = read_3D_buffer(q, b.community_state, p.N_graphs, {acc_event});
//     state_df.resize_dim(2, 1);
//     write_dataframe(p.output_dir + "/Graph_", "/Trajectories/community_trajectory_", state_df, false);
// }

void p_I_run(sycl::queue &q, Multiple_Sim_Param_t p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcm, const Dataframe_t<float, 3> &p_Is)
{
    auto p_I_lin = p_Is.flatten();
    auto b = Sim_Buffers::make(q, p.to_sim_param(), edge_list, vcm, p_I_lin);
    b.validate_sizes(p.to_sim_param());
    run(q, p, b);
}

void multiple_sim_param_run(sycl::queue &q, const Multiple_Sim_Param_t &p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 3> &edge_list, const Dataframe_t<uint32_t, 3> &vcm, Dataframe_t<float, 4> p_Is)
{
    if (p_Is.size() == 0)
    {
        auto N_p_out = p.p_out.size();
        auto N_connections_max = complete_graph_size(p.N_communities);
        p_Is.data.resize(N_p_out);
        for (int i = 0; i < N_p_out; i++)
        {
            p_Is.data[i] = generate_duplicated_p_Is(p.Nt, p.N_sims * p.N_graphs, N_connections_max, p.p_I_min[i], p.p_I_max[i], p.seed);
        }
    }

    p_I_run(q, p, edge_list.flatten_dim_0(), vcm.flatten_dim_0(), p_Is.flatten_dim_0());
}
