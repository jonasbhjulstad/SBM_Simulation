#include <Sycl_Graph/Database/Dataframe.hpp>
#include <Sycl_Graph/Database/Simulation_Tables.hpp>
#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <Sycl_Graph/Simulation/State_Accumulation.hpp>
#include <chrono>

void write_allocated_steps(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, pqxx::connection &con, uint32_t t, std::vector<sycl::event> &dep_events)
{
    auto N_steps = t % p.Nt_alloc;
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
    auto df_range = state_df.get_ranges();
    write_graphseries(con, p.p_out_idx, state_df, "community_state", t);
    write_graphseries(con, p.p_out_idx, event_df, "connection_events", t);

    auto inf_gs = sample_infections(state_df, event_df, b.ccm, p.seed);
    write_graphseries(con, p.p_out_idx, inf_gs, "infection_events", t);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Inf sample/ write graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
}

void write_initial_steps(sycl::queue &q, const Sim_Param &p, pqxx::connection& con, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
{
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, p.compute_range, p.wg_range, p.N_sims);
    auto state_df = read_3D_buffer(q, b.community_state, p.N_graphs, {acc_event});
    state_df.resize_dim(2, 1);
    write_graphseries(con, p.p_out_idx, state_df, "community_state");
}

void run(sycl::queue &q, Sim_Param p,pqxx::connection &con, Sim_Buffers &b)
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
    write_initial_steps(q, p, con, b, events);
    uint32_t t = 0;
    for (t = 0; t < p.Nt; t++)
    {
        bool is_initial_write = (t == 0);
        if (is_allocated_space_full(t, p.Nt_alloc))
        {
            q.wait();
            write_allocated_steps(q, p, b, con, t, events);
            events[0] = clear_buffer<uint32_t, 3>(q, b.accumulated_events, events);
            events[0] = move_buffer_row(q, b.vertex_state, p.Nt_alloc, events);
        }
        events = recover(q, p, b.vertex_state, b.rngs, t, events);
        events = infect(q, p, b, t, events);
        std::cout << "t: " << t << "\n";
    }
    write_allocated_steps(q, p, b,con, t, events);

}

void run(sycl::queue &q, Sim_Param p, pqxx::connection& con, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcm)
{

    auto b = Sim_Buffers::make(q, p, con, edge_list, vcm, {});
    b.validate_sizes(p);
    run(q, p, b);
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
