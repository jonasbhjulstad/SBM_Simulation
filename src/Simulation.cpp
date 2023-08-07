
#include <Sycl_Graph/Simulation.hpp>

#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/SIR_Dynamics.hpp>
#include <Sycl_Graph/SIR_Infection_Sampling.hpp>
#include <algorithm>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

std::vector<uint32_t> count_ecm(const std::vector<uint32_t> &ecm)
{
    uint32_t max_idx = *std::max_element(ecm.begin(), ecm.end()) + 1;
    std::vector<uint32_t> result(max_idx, 0);
    for (int i = 0; i < ecm.size(); i++)
    {
        result[ecm[i]]++;
    }
    return result;
}

void simulation_param_to_file(const Sim_Param &p, std::ofstream &log_file)
{
    log_file << "Nt: " << p.Nt << std::endl;
    log_file << "N_clusters: " << p.N_clusters << std::endl;
    log_file << "N_pop: " << p.N_pop << std::endl;
    log_file << "p_in: " << p.p_in << std::endl;
    log_file << "p_out: " << p.p_out << std::endl;
    log_file << "p_R0: " << p.p_R0 << std::endl;
    log_file << "p_I0: " << p.p_I0 << std::endl;
    log_file << "p_R: " << p.p_R << std::endl;
    log_file << "seed: " << p.seed << std::endl;
    log_file << "sim_idx: " << p.sim_idx << std::endl;
    log_file << "N_pop_tot: " << p.N_pop << std::endl;
    log_file << "max_infection_samples: " << p.max_infection_samples << std::endl;
}

void buffer_sizes_to_file(
    const auto &edge_from_buf,
    const auto &edge_to_buf,
    const auto &ecm_buf,
    const auto &vcm_buf,
    const auto &p_I_buf,
    const auto &event_to_buf,
    const auto &event_from_buf,
    const auto &community_state_buf,
    const auto &trajectory_buf,
    std::ofstream &log_file)
{
    log_file << "edge_from_buf: " << edge_from_buf.size() << std::endl;
    log_file << "edge_to_buf: " << edge_to_buf.size() << std::endl;
    log_file << "ecm_buf: " << ecm_buf.size() << std::endl;
    log_file << "vcm_buf: " << vcm_buf.size() << std::endl;
    log_file << "p_I_buf: " << p_I_buf.get_range()[0] << ", " << p_I_buf.get_range()[1] << std::endl;
    log_file << "event_to_buf: " << event_to_buf.get_range()[0] << ", " << event_to_buf.get_range()[1] << std::endl;
    log_file << "event_from_buf: " << event_from_buf.get_range()[0] << ", " << event_from_buf.get_range()[1] << std::endl;
    log_file << "community_state_buf: " << community_state_buf.get_range()[0] << ", " << community_state_buf.get_range()[1] << std::endl;
    log_file << "trajectory_buf: " << trajectory_buf.get_range()[0] << ", " << trajectory_buf.get_range()[1] << std::endl;
}

std::vector<std::vector<uint32_t>> column_zip_2D(const std::vector<std::vector<uint32_t>> &vec0, const std::vector<std::vector<uint32_t>> &vec1)
{
    uint32_t rows = vec0.size();
    uint32_t cols = vec0[0].size();
    std::vector<std::vector<uint32_t>> result(rows, std::vector<uint32_t>(cols * 2));
    assert(std::all_of(vec0.begin(), vec0.end(), [cols](const auto &v)
                       { return v.size() == cols; }));
    assert(std::all_of(vec1.begin(), vec1.end(), [cols](const auto &v)
                       { return v.size() == cols; }));

    for (int i = 0; i < rows; i++)
    {
        std::copy(vec0[i].begin(), vec0[i].end(), result[i].begin());
        std::copy(vec1[i].begin(), vec1[i].end(), result[i].begin() + cols);
    }
    return result;
}

Sim_Data excite_simulate(const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir, bool debug_flag, sycl::queue q)
{
    // forwarding
    uint32_t Nt = p.Nt;
    uint32_t N_clusters = p.N_clusters;
    uint32_t N_pop = p.N_pop;
    float p_in = p.p_in;
    float p_out = p.p_out;
    float p_R0 = p.p_R0;
    float p_I0 = p.p_I0;
    float p_R = p.p_R;
    uint32_t seed = p.seed;
    uint32_t sim_idx = p.sim_idx;
    uint32_t N_pop_tot = N_pop * N_clusters;
    std::ofstream log_file;
    if (debug_flag)
    {
        std::filesystem::create_directories(output_dir);
        log_file.open((output_dir + "/Simulation_Debug") + std::to_string(p.sim_idx) + ".log");
    }

    if (debug_flag)
    {
        std::cout << "Initializing Simulation" << std::endl;
        simulation_param_to_file(p, log_file);
    }
    auto device = q.get_device();
    // get work group size
    auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();
    if (debug_flag)
        std::cout << "Work group size: " << N_wg << std::endl;

    auto ecm = ecm_from_vcm(edge_list, vcm);
    uint32_t N_connections = std::max_element(ecm.begin(), ecm.end())[0] + 1;

    auto ccm = complete_ccm(p.N_clusters);
    auto ccm_weights = count_ecm(ecm);
    if (debug_flag)
        std::cout << "Initializing Buffers" << std::endl;

    std::vector<uint32_t> edge_from_init(edge_list.size());
    std::vector<uint32_t> edge_to_init(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), edge_from_init.begin(), [](auto &e)
                   { return e.first; });
    std::transform(edge_list.begin(), edge_list.end(), edge_to_init.begin(), [](auto &e)
                   { return e.second; });
    std::vector<std::vector<uint32_t>> connection_events_init(Nt, std::vector<uint32_t>(N_connections, 0));
    std::vector<std::vector<State_t>> community_state_init(Nt + 1, std::vector<State_t>(N_clusters, {0, 0, 0}));
    std::vector<std::vector<float>> p_I_vec = generate_p_Is(N_connections, p_I_min, p_I_max, Nt, seed);
    auto seed_buf = generate_seeds(q, N_wg, seed);
    auto trajectory = sycl::buffer<SIR_State, 2>(sycl::range<2>(Nt + 1, N_pop_tot));
    std::vector<sycl::event> b_events(8);

    auto edge_from_buf = buffer_create_1D(q, edge_from_init, b_events[0]);
    auto edge_to_buf = buffer_create_1D(q, edge_to_init, b_events[1]);
    auto ecm_buf = buffer_create_1D(q, ecm, b_events[2]);
    auto vcm_buf = buffer_create_1D(q, vcm, b_events[3]);
    auto p_I_buf = buffer_create_2D(q, p_I_vec, b_events[4]);
    auto event_to_buf = buffer_create_2D(q, connection_events_init, b_events[5]);
    auto event_from_buf = buffer_create_2D(q, connection_events_init, b_events[6]);
    auto community_state_buf = buffer_create_2D(q, community_state_init, b_events[7]);
    auto advance_event = initialize_vertices(p_I0, p_R0, q, trajectory, seed_buf, b_events);

    if (debug_flag)
        buffer_sizes_to_file(edge_from_buf, edge_to_buf, ecm_buf, vcm_buf, p_I_buf, event_to_buf, event_from_buf, community_state_buf, trajectory, log_file);

    uint32_t t = 0;

    sycl::event rec_event;
    sycl::event inf_event;
    if (debug_flag)
        std::cout << "Enqueueing Kernels" << std::endl;

    for (int t = 0; t < Nt; t++)
    {
        auto rec_event = recover(q, t, inf_event, p_R, seed_buf, trajectory, vcm_buf);

        inf_event = infect(q, ecm_buf, p_I_buf, seed_buf, event_from_buf, event_to_buf, trajectory, edge_from_buf, edge_to_buf, N_wg, t, N_connections, rec_event);
    }

    Sim_Data result(trajectory, event_from_buf, event_to_buf);
    result.trajectory = trajectory;
    result.event = inf_event;
    result.output_dir = output_dir;
    result.seed = seed;
    result.sim_idx = sim_idx;
    result.ccm = ccm;
    result.ccm_weights = ccm_weights;
    result.p_I_vec = p_I_vec;
    result.vcm = vcm;
    return result;
}

void excite_simulation_read_to_files(const Sim_Param &p, sycl::queue &q, Sim_Data &d, bool debug_flag)
{
    uint32_t N_clusters = p.N_clusters;
    uint32_t Nt = d.trajectory.get_range()[0] - 1;
    std::ofstream log_file;
    if (debug_flag)
        log_file.open(d.output_dir + "Simulation_Debug.log");
    std::cout << "Reading Buffers" << std::endl;

    auto vertex_state = read_buffer(q, d.trajectory, d.event, log_file);

    std::vector<std::vector<State_t>> community_state(Nt + 1, std::vector<State_t>(N_clusters, {0, 0, 0}));
    std::transform(std::execution::par_unseq, vertex_state.begin(), vertex_state.end(), community_state.begin(), [=](auto v_state)
                   {

        std::vector<State_t> state(N_clusters, {0, 0, 0});
        for(int i = 0; i < v_state.size(); i++)
        {
            auto community_idx = d.vcm[i];
            state[community_idx][v_state[i]]++;
        }
        return state; });

    auto from_events = read_buffer(q, d.event_from_buf, d.event, log_file);
    auto to_events = read_buffer(q, d.event_to_buf, d.event, log_file);

    auto connection_infections = sample_infections(community_state, from_events, to_events, d.ccm, d.ccm_weights, d.seed, p.max_infection_samples);

    auto connection_events = column_zip_2D(from_events, to_events);

    if (debug_flag)
        std::cout << "Writing to directory: " << d.output_dir << std::endl;

    std::filesystem::create_directories(d.output_dir);

    std::ofstream community_traj_f(d.output_dir + "community_trajectory_" +
                                   std::to_string(d.sim_idx) + ".csv");
    std::ofstream connection_events_f(d.output_dir + "connection_events_" +
                                      std::to_string(d.sim_idx) + ".csv");

    std::ofstream connection_infections_f(d.output_dir + "connection_infections_" +
                                          std::to_string(d.sim_idx) + ".csv");

    std::for_each(community_state.begin(), community_state.end(),
                  [&](auto &community_trajectory_i)
                  {
                      linewrite(community_traj_f, community_trajectory_i);
                  });
    std::for_each(connection_events.begin(),
                  connection_events.end(),
                  [&](auto &connection_events_i)
                  {
                      linewrite(connection_events_f, connection_events_i);
                  });

    std::for_each(connection_infections.begin(),
                  connection_infections.end(),
                  [&](auto &connection_infections_i)
                  {
                      linewrite(connection_infections_f, connection_infections_i);
                  });
    std::ofstream p_I_f(d.output_dir + "p_Is_" + std::to_string(d.sim_idx) + ".csv");
    std::for_each(d.p_I_vec.begin(), d.p_I_vec.end(),
                  [&](auto &p_I_i)
                  { linewrite(p_I_f, p_I_i); });

    std::ofstream ccm_f(d.output_dir + "ccm_" + std::to_string(d.sim_idx) + ".csv");
    std::for_each(d.ccm.begin(), d.ccm.end(),
                  [&](const auto &ccm_i)
                  { ccm_f << ccm_i.first << "," << ccm_i.second << "\n"; });
}

void parallel_excite_simulate(const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir, uint32_t N_simulations, sycl::queue q, bool debug_flag)
{
    auto log_dirname = [output_dir](auto idx)
    { return (output_dir + "/Graph_") + std::to_string(idx) + "/"; };
    std::mt19937_64 rng(p.seed);

    std::vector<Sim_Param> sim_params(N_simulations, p);
    for (int i = 0; i < N_simulations; i++)
    {
        sim_params[i].seed = rng();
        sim_params[i].sim_idx = i;
    }
    std::vector<Sim_Data> simdata(N_simulations);

    std::transform(std::execution::par_unseq, sim_params.begin(), sim_params.end(), simdata.begin(), [&](const Sim_Param &p)
                   { if(debug_flag)
                    std::cout << "Enqueuing Simulation " << p.sim_idx << std::endl;
                    return excite_simulate(p, vcm, edge_list, p_I_min, p_I_max, log_dirname(p.sim_idx), 0, q); });

    std::for_each(std::execution::par_unseq, simdata.begin(), simdata.end(), [&](auto &d)
                  { if (debug_flag)
                    std::cout << "Reading Simulation " << d.sim_idx << std::endl;
                    excite_simulation_read_to_files(p, q, d, 0); });
}
