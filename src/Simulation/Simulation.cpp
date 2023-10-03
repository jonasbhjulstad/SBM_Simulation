#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <chrono>
#include <Sycl_Graph/Simulation/State_Accumulation.hpp>


void write_allocated_steps(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, size_t N_steps, std::vector<sycl::event> &dep_events)
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
    for (int i = 0; i < p.p_out_vec.size(); i++)
    {

        auto p_out = p.p_out_vec[i];
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

void write_initial_steps(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
{
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range, p.N_sims);
    auto state_df = read_3D_buffer(q, b.community_state, p.N_graphs, {acc_event});
    state_df.resize_dim(2, 1);
    write_dataframe(p.output_dir + "/Graph_", "/Trajectories/community_trajectory_", state_df, false);
}

void reset_directories(const Sim_Param &p)
{
    for (int i = 0; i < p.N_graphs; i++)
    {
        std::filesystem::remove_all(p.output_dir + "/Graph_" + std::to_string(i) + "/");
    }
    std::filesystem::create_directories(p.output_dir);
    // create directories for all p.N_graphs
    for (int i = 0; i < p.N_graphs; i++)
    {
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/Trajectories/");
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/p_Is/");
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/Connection_Infections/");
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/Connection_Events/");
    }
}

void run(sycl::queue &q, Sim_Param p, Sim_Buffers &b)
{
    if ((p.global_mem_size == 0) || p.local_mem_size == 0)
    {
        auto device = q.get_device();
        p.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        p.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    }

    std::vector<sycl::event> events(1);
    q.wait();
    events[0] = initialize_vertices(q, p, b.vertex_state, b.rngs);
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
        events = recover(q, p, b.vertex_state, b.rngs, t, events);
        events = infect(q, p, b, t, events);
        std::cout << "t: " << t << "\n";
    }
    write_allocated_steps(q, p, b, t % p.Nt_alloc, events);

    if_false_throw(b.ccm.size() == p.N_graphs, "ccm size does not match N_graphs");
    write_dataframe(p.output_dir + "/Graph_", "/ccm.csv", b.ccm, false);

    auto p_I_df = read_3D_buffer(q, b.p_Is, p.N_graphs, events);
    write_dataframe(p.output_dir + "/Graph_", "/p_Is/p_I_", p_I_df, true);
    p.dump(p.output_dir + "/Sim_Param.json");
}

void run(sycl::queue &q, Sim_Param p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcm)
{

    auto b = Sim_Buffers::make(q, p, edge_list, vcm, {});
    b.validate_sizes(p);
    run(q, p, b);
    for (int graph_idx = 0; graph_idx < edge_list.size(); graph_idx++)
    {
        write_edgelist(p.output_dir + "/Graph_" + std::to_string(graph_idx) + "/edgelist.csv", edge_list[graph_idx]);
        write_vector(p.output_dir + "/Graph_" + std::to_string(graph_idx) + "/vcm.csv", vcm[graph_idx]);
        // write_vector(p.output_dir + "/Graph_" + std::to_string(graph_idx) + "/ecm.csv", ecm[graph_idx]);
    }
}

void run(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm)
{
    auto edge_list_df = Dataframe_t<std::pair<uint32_t, uint32_t>, 2>(edge_list);
    auto vcm_df = Dataframe_t<uint32_t, 2>(vcm);
    run(q, p, edge_list_df, vcm_df);
}


auto matrix_linearize(const std::vector<std::vector<float>> &vecs)
{
    std::vector<float> out;
    out.reserve(vecs.size() * vecs[0].size());
    for (auto &&v : vecs)
    {
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}


void p_I_run(sycl::queue &q, Sim_Param p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcm, const Dataframe_t<float, 3> &p_Is)
{
    auto p_I_lin = p_Is.flatten();
    auto b = Sim_Buffers::make(q, p, edge_list, vcm, p_I_lin);
    b.validate_sizes(p);
    run(q, p, b);
}

void p_I_run(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm, const std::vector<std::vector<std::vector<float>>> &p_Is)
{
    // create dataframes
    auto edge_list_df = Dataframe_t<std::pair<uint32_t, uint32_t>, 2>(edge_list);
    auto vcm_df = Dataframe_t<uint32_t, 2>(vcm);
    auto p_Is_df = Dataframe_t<float, 3>(p_Is);
    p_I_run(q, p, edge_list_df, vcm_df, p_Is_df);
}
