#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <SBM_Database/SBM_Database.hpp>
#include <chrono>

Simulation_t::Simulation_t(sycl::queue &q, soci::session &sql, const Sim_Param &sim_param, const Dataframe::Dataframe_t<Edge_t, 2> &edge_list, const Dataframe::Dataframe_t<uint32_t, 2> &vcm, sycl::range<1> compute_range, sycl::range<1> wg_range, const Dataframe::Dataframe_t<float, 3> &p_Is)
    : q(q), sql(sql), p(sim_param), b(q, sim_param, sql, edge_list, vcm, p_Is), compute_range(compute_range), wg_range(wg_range)
{
}

Simulation_t::Simulation_t(sycl::queue &q, soci::session &sql, const Sim_Param &sim_param, const Sim_Buffers &sim_buffers, sycl::range<1> compute_range, sycl::range<1> wg_range)
    : q(q), sql(sql), p(sim_param), b(sim_buffers), compute_range(compute_range), wg_range(wg_range)
{
}

void Simulation_t::write_allocated_steps(uint32_t t, std::vector<sycl::event> &dep_events, uint32_t N_max_steps)
{
    auto N_steps = t % p.Nt_alloc;
    N_steps = (N_steps == 0) ? p.Nt_alloc : N_steps;
    N_steps = (N_max_steps) ? std::min<uint32_t>({N_steps, N_max_steps}) : N_steps;
    std::chrono::high_resolution_clock::time_point t1, t2;
    N_steps = std::min<size_t>({N_steps, p.Nt_alloc});
    t1 = std::chrono::high_resolution_clock::now();
    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, compute_range, wg_range, p.N_sims);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Accumulate community state: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    auto state_df = read_3D_buffer(q, *b.community_state, p.N_graphs, {acc_event});
    auto event_df = read_3D_buffer(q, *b.accumulated_events, p.N_graphs, {acc_event})({0, 0, 0, 0}, {p.N_graphs, p.N_sims, N_steps, p.N_connections_max()});
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Read graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    auto df_range = state_df.get_ranges();


// void dataframe_insert(soci::session &sql, const Dataframe::Dataframe_t<T, N_df> &df,
//                         const std::string &table_name,
//                         const std::array<std::string, N_const> &constant_indices,
//                         const std::array<uint32_t, N_const> &constant_index_values,
//                         const std::array<std::string, N_df> &iterable_index_names,
//                         const std::string &data_name)

    SBM_Database::dataframe_insert<State_t, 4, 1>(sql, state_df, "community_state", {"graph", "sim", "t", "community"}, "state", {"p_out"}, {p.p_out_idx});
    SBM_Database::dataframe_insert<uint32_t, 4, 1>(sql, event_df, "connection_events", {"graph", "sim", "t", "connection"}, "event", {"p_out"}, {p.p_out_idx});

    // SBM_Database::write_graphseries(sql, p.p_out_idx, state_df, "community_state", t - p.Nt_alloc);
    // SBM_Database::write_graphseries(sql, p.p_out_idx, event_df, "connection_events", t - p.Nt_alloc);
    state_df.resize_dim(2, N_steps + 1);
    event_df.resize_dim(2, N_steps);
    auto inf_gs = sample_infections(state_df, event_df, b.ccm, p.seed);
    // SBM_Database::write_graphseries(sql, p.p_out_idx, inf_gs, "infection_events", t - p.Nt_alloc);
    SBM_Database::dataframe_insert<uint32_t, 4, 1>(sql, inf_gs, "infection_events", {"graph", "sim", "t", "community"}, {"event"}, {"p_out"}, {p.p_out_idx});
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Inf sample/ write graphseries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
}

void Simulation_t::write_initial_steps(sycl::queue &q, const Sim_Param &p, soci::session &sql, Sim_Buffers &b, std::vector<sycl::event> &dep_events)
{

    auto acc_event = accumulate_community_state(q, dep_events, b.vertex_state, b.vcm, b.community_state, compute_range, wg_range, p.N_sims);
    auto state_df = read_3D_buffer(q, *b.community_state, p.N_graphs, {acc_event});
    state_df.resize_dim(2, 1);
    SBM_Database::dataframe_insert<State_t, 4, 1>(sql, state_df, "community_state", {"graph", "sim", "t", "community"}, {"state"}, {"p_out"}, {p.p_out_idx});
    // SBM_Database::write_graphseries(sql, p.p_out_idx, state_df, "community_state");
}

void Simulation_t::run()
{
    if ((!compute_range[0]) || (!wg_range[0]))
    {
        std::tie(compute_range, wg_range) = default_compute_range(q);
    }
    std::vector<sycl::event> events(1);
    // q.wait();
    // read_vcm(q, b.vcm);
    q.wait();

    // events[0] = initialize_vertices(q, p, b.vertex_state, b.rngs, compute_range, wg_range, b.construction_events);
    write_initial_steps(q, p, sql, b, events);
    uint32_t t = 0;
    for (t = 0; t < p.Nt; t++)
    {
        bool is_initial_write = (t == 0);
        if (is_allocated_space_full(t, p.Nt_alloc))
        {
            q.wait();
            write_allocated_steps(t, events);
            events[0] = clear_buffer<uint32_t, 3>(q, *b.accumulated_events, events);
            events[0] = move_buffer_row(q, b.vertex_state, p.Nt_alloc, events);
        }
        events = recover(q, p, *b.vertex_state, *b.rngs, t, compute_range, wg_range, events);
        events = infect(q, p, b, t, compute_range, wg_range, events);
        std::cout << "t: " << t << "\n";
    }
    write_allocated_steps(t, events);
}

std::tuple<sycl::range<1>, sycl::range<1>> Simulation_t::default_compute_range(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
    auto max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    auto N_wg = std::min<uint32_t>({max_compute_units, (uint32_t)max_work_group_size, (uint32_t)p.N_sims_tot()});

    auto N_compute = static_cast<uint32_t>(std::ceil(static_cast<float>(p.N_sims * p.N_graphs) / static_cast<float>(N_wg))) * N_wg;

    return std::make_tuple(sycl::range<1>(N_compute), sycl::range<1>(N_wg));
}
