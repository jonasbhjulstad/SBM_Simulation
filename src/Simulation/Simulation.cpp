#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <chrono>

void single_community_state_accumulate(sycl::nd_item<1> &it, const auto &vcm_acc, const sycl::accessor<SIR_State, 3, sycl::access_mode::read> &v_acc, const sycl::accessor<State_t, 3, sycl::access_mode::read_write> &state_acc, uint32_t N_sims)
{
    auto Nt = v_acc.get_range()[0];
    auto N_vertices = v_acc.get_range()[2];
    auto sim_id = it.get_global_id()[0];
    auto graph_id = static_cast<uint32_t>(std::floor(static_cast<float>(sim_id) / static_cast<float>(N_sims)));
    auto N_communities = state_acc.get_range()[2];
    for (int t = 0; t < Nt; t++)
    {
        for (int c_idx = 0; c_idx < N_communities; c_idx++)
        {
            state_acc[t][sim_id][c_idx] = {0, 0, 0};
        }
        for (int v_idx = 0; v_idx < N_vertices; v_idx++)
        {
            auto c_idx = vcm_acc[graph_id][v_idx];
            auto v_state = v_acc[t][sim_id][v_idx];
            state_acc[t][sim_id][c_idx][v_state]++;
        }
    }
}

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims)
{
    auto Nt = v_buf.get_range()[0];
    auto range = sycl::range<3>(Nt, v_buf.get_range()[1], v_buf.get_range()[2]);
    return q.submit([&](sycl::handler &h)
                    {
                h.depends_on(dep_events);
        auto v_acc = construct_validate_accessor<SIR_State, 3, sycl::access_mode::read>(v_buf, h, range);
        sycl::accessor<State_t, 3, sycl::access_mode::read_write> state_acc(community_buf, h);
        auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read>(h);

            h.parallel_for(sycl::nd_range<1>(compute_range, wg_range), [=](sycl::nd_item<1> it)
            {

                single_community_state_accumulate(it, vcm_acc, v_acc, state_acc, N_sims);
            }); });
}

sycl::event move_buffer_row(sycl::queue &q, sycl::buffer<SIR_State, 3> &buf, uint32_t row, std::vector<sycl::event> &dep_events)
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

    write_dataframe(p.output_dir + "/Graph_", p.simulation_subdir + "Trajectories/community_trajectory_", state_df_write, true, {1, 0});
    write_dataframe(p.output_dir + "/Graph_", p.simulation_subdir + "Connection_Events/connection_events_", event_df, true);
    // event_inf_summary(state_df, event_df, b.ccm);

    auto inf_gs = sample_infections(state_df, event_df, b.ccm, p.seed);
    write_dataframe(p.output_dir + "/Graph_", p.simulation_subdir + "Connection_Infections/connection_infections_", inf_gs, true);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Inf sample/ write graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
}

void write_initial_steps(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
{
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range, p.N_sims);
    auto state_df = read_3D_buffer(q, b.community_state, p.N_graphs, {acc_event});
    state_df.resize_dim(2, 1);
    write_dataframe(p.output_dir + "/Graph_", p.simulation_subdir + "Trajectories/community_trajectory_", state_df, false);
}

void reset_directories(const Sim_Param &p)
{
    for(int i = 0; i < p.N_graphs; i++)
    {
    std::filesystem::remove_all(p.output_dir + "/Graph_" + std::to_string(i) + "/" + p.simulation_subdir);
    }
    std::filesystem::create_directories(p.output_dir);
    // create directories for all p.N_graphs
    for (int i = 0; i < p.N_graphs; i++)
    {
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/" + p.simulation_subdir + "Trajectories/");
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/" + p.simulation_subdir + "p_Is/");
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/" + p.simulation_subdir + "Connection_Infections/");
        std::filesystem::create_directories(p.output_dir + "/Graph_" + std::to_string(i) + "/" + p.simulation_subdir + "Connection_Events/");
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
    write_dataframe(p.output_dir + "/Graph_",  p.simulation_subdir + "/ccm.csv", b.ccm, false);

    auto p_I_df = read_3D_buffer(q, b.p_Is, p.N_graphs, events);
    write_dataframe(p.output_dir + "/Graph_", p.simulation_subdir + "p_Is/p_I_", p_I_df, true);
    p.dump(p.output_dir + "/Sim_Param.json");
}

void run(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm)
{

    auto b = Sim_Buffers::make(q, p, edge_list, vcm, {});
    b.validate_sizes(p);
    run(q, p, b);
    for (int graph_idx = 0; graph_idx < edge_list.size(); graph_idx++)
    {
        write_edgelist(p.output_dir + "/Graph_" + std::to_string(graph_idx) + p.simulation_subdir + "/edgelist.csv", edge_list[graph_idx]);
        write_vector(p.output_dir + "/Graph_" + std::to_string(graph_idx) + p.simulation_subdir + "/vcm.csv", vcm[graph_idx]);
        // write_vector(p.output_dir + "/Graph_" + std::to_string(graph_idx) + "/ecm.csv", ecm[graph_idx]);
    }
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

auto dataframe_linearize(const std::vector<std::vector<std::vector<float>>> &df)
{
    std::vector<float> result(df.size() * df[0].size() * df[0][0].size());
    for (int i = 0; i < df.size(); i++)
    {
        for (int j = 0; j < df[0].size(); j++)
        {
            for (int k = 0; k < df[0][0].size(); k++)
            {
                result[i * df[0].size() * df[0][0].size() + j * df[0][0].size() + k] = df[i][j][k];
            }
        }
    }
    return result;
}

void p_I_run(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm, const std::vector<std::vector<std::vector<float>>> &p_Is)
{
    auto p_I_lin = dataframe_linearize(p_Is);
    auto b = Sim_Buffers::make(q, p, edge_list, vcm, p_I_lin);
    b.validate_sizes(p);
    run(q, p, b);
}
