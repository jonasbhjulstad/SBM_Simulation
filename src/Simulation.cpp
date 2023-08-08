
#include <Sycl_Graph/Simulation.hpp>

#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/SIR_Dynamics.hpp>
#include <Sycl_Graph/SIR_Infection_Sampling.hpp>
#include <algorithm>
#include <chrono>
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
    const auto &events_to_buf,
    const auto &events_from_buf,
    const auto &community_state_buf,
    const auto &trajectory_buf,
    std::ofstream &log_file)
{
    log_file << "edge_from_buf: " << edge_from_buf->size() << std::endl;
    log_file << "edge_to_buf: " << edge_to_buf->size() << std::endl;
    log_file << "ecm: " << ecm_buf->size() << std::endl;
    log_file << "vcm_buf: " << vcm_buf->size() << std::endl;
    log_file << "p_I_buf: " << p_I_buf.get_range()[0] << ", " << p_I_buf.get_range()[1] << std::endl;
    log_file << "events_to_buf: " << events_to_buf.get_range()[0] << ", " << events_to_buf.get_range()[1] << std::endl;
    log_file << "events_from_buf: " << events_from_buf.get_range()[0] << ", " << events_from_buf.get_range()[1] << std::endl;
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
    auto ecm = ecm_from_vcm(edge_list, vcm);
    uint32_t N_connections = std::max_element(ecm.begin(), ecm.end())[0] + 1;
    auto p_I_vec = generate_p_Is(N_connections, p_I_min, p_I_max, p.Nt, p.seed);
    return fixed_simulate(q, p, edge_list, vcm, p_I_vec, output_dir, debug_flag);
}

struct Simulation_Buffers
{
    Simulation_Buffers(sycl::queue &q, const std::vector<uint32_t> &edge_from_init, const std::vector<uint32_t> &edge_to_init, const std::vector<uint32_t> &ecm_init, const std::vector<uint32_t> &vcm_init, uint32_t N_clusters, uint32_t N_connections, uint32_t Nt, uint32_t seed) : edge_from{shared_buffer_create_1D(q, edge_from_init, events[0])},
                                                                                                                                                                                                                                                                                        edge_to{shared_buffer_create_1D(q, edge_to_init, events[1])},
                                                                                                                                                                                                                                                                                        ecm{shared_buffer_create_1D(q, ecm_init, events[2])},
                                                                                                                                                                                                                                                                                        vcm{shared_buffer_create_1D(q, vcm_init, events[3])}, N_connections(N_connections)
    {
    }
    Simulation_Buffers(const auto p_edge_from, const auto p_edge_to, const auto p_ecm, const auto p_vcm, const auto &ccm, const auto &ccm_weights, const std::vector<sycl::event> &events, uint32_t N_connections)
        : edge_from{p_edge_from}, edge_to{p_edge_to}, ecm{p_ecm}, vcm{p_vcm}, ccm{ccm}, ccm_weights{ccm_weights}, events(events), N_connections(N_connections)
    {
    }
    Simulation_Buffers(const Simulation_Buffers &other) : Simulation_Buffers(other.edge_from, other.edge_to, other.ecm, other.vcm, other.ccm, other.ccm_weights, other.events, other.N_connections) {}

    std::vector<sycl::event> events = std::vector<sycl::event>(4);
    std::shared_ptr<sycl::buffer<uint32_t>> edge_from;
    std::shared_ptr<sycl::buffer<uint32_t>> edge_to;
    std::shared_ptr<sycl::buffer<uint32_t>> ecm;
    std::shared_ptr<sycl::buffer<uint32_t>> vcm;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::vector<uint32_t> ccm_weights;
    uint32_t N_connections;
};
Sim_Data enqueue_kernels(sycl::queue &q, const Sim_Param &p, const Simulation_Buffers &d,
                         auto &seed_buf, auto &events_from_buf, auto &events_to_buf, auto &community_state_buf, auto &trajectory_buf, auto& p_I_buf, const std::vector<uint32_t>& ccm,
                         const std::string &output_dir, bool debug_flag, sycl::event &dep_event)
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
    std::ofstream log_file;

    auto& edge_from_buf = d.edge_from;
    auto& edge_to_buf = d.edge_to;
    auto& ecm_buf = d.ecm;
    auto& vcm_buf = d.vcm;
    auto& ccm_weights = d.ccm_weights;
    uint32_t N_edges = ecm_buf->size();
    uint32_t N_connections = d.N_connections;
    std::chrono::time_point<std::chrono::steady_clock> start, end;

    if (debug_flag)
    {
        std::filesystem::create_directories(output_dir);
        log_file.open((output_dir + "/Simulation_Debug") + std::to_string(p.sim_idx) + ".log");
        std::cout << "Initializing Simulation" << std::endl;
        simulation_param_to_file(p, log_file);
        std::cout << "Initializing Buffers" << std::endl;
        start = std::chrono::steady_clock::now();
        buffer_sizes_to_file(d.edge_from, d.edge_to, ecm_buf, vcm_buf, p_I_buf, events_to_buf, events_from_buf, community_state_buf, trajectory_buf, log_file);
    }
    std::vector<sycl::event> b_events(9);

    uint32_t t = 0;

    sycl::event rec_event;
    sycl::event inf_event;
    if (debug_flag)
        std::cout << "Enqueueing Kernels" << std::endl;
    sycl::buffer<uint32_t> infection_indices_buf((sycl::range<1>(N_edges)));
    for (int t = 0; t < Nt; t++)
    {
        auto rec_event = recover(q, t, dep_event, p_R, seed_buf, trajectory_buf);

        inf_event = infect(q, ecm_buf, p_I_buf, seed_buf, events_from_buf, events_to_buf, trajectory_buf, edge_from_buf, edge_to_buf, infection_indices_buf, t, N_connections, rec_event);
    }

    Sim_Data result(trajectory_buf, events_from_buf, events_to_buf);
    result.event = inf_event;
    result.output_dir = output_dir;
    result.seed = p.seed;
    result.sim_idx = p.sim_idx;
    result.ccm_weights = ccm_weights;
    result.ccm = ccm;
    if (debug_flag)
        {
            end = std::chrono::steady_clock::now();
            log_file << "Enqueue time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        }
    return result;
}

Sim_Data fixed_simulate(sycl::queue &q, const Sim_Param &p, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &vcm, const std::vector<std::vector<float>> &p_I_vec, const std::string &output_dir, bool debug_flag)
{
    auto ecm = ecm_from_vcm(edge_list, vcm);
    uint32_t N_connections = std::max_element(ecm.begin(), ecm.end())[0] + 1;

    std::ofstream log_file;
    std::chrono::steady_clock::time_point begin, end;
    if (debug_flag)
    {
        log_file.open(output_dir + "/simulation_buffer_init_" + std::to_string(p.sim_idx) + ".log");
        begin = std::chrono::steady_clock::now();
    }
    //Initialize Common Buffers
    std::vector<uint32_t> edge_from_init(edge_list.size());
    std::vector<uint32_t> edge_to_init(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), edge_from_init.begin(), [](auto &e)
                   { return e.first; });
    std::transform(edge_list.begin(), edge_list.end(), edge_to_init.begin(), [](auto &e)
                   { return e.second; });
    Simulation_Buffers data(q, edge_from_init, edge_to_init, ecm, vcm, p.N_clusters, N_connections, p.Nt, p.seed);
    auto device = q.get_device();


    // get work group size
    auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();
    if (debug_flag)
        std::cout << "Work group size: " << N_wg << std::endl;
    auto ccm = complete_ccm(p.N_clusters);
    auto ccm_weights = count_ecm(ecm);

    //Write to file
    std::ofstream p_I_f(output_dir + "/p_Is_" + std::to_string(p.sim_idx) + ".csv");
    std::for_each(p_I_vec.begin(), p_I_vec.end(), [&](const auto& pv)
    {
        linewrite(p_I_f, pv);
    });

    std::ofstream ccm_f(output_dir + "/ccm.csv");
    std::for_each(ccm.begin(), ccm.end(), [&](const auto& p){ccm_f << p.first << "," << p.second << std::endl;});
    if (debug_flag)
    {
        end = std::chrono::steady_clock::now();
        log_file << "Common buffers construction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
        begin = std::chrono::steady_clock::now();
    }

    //Initialize individual buffers

    std::vector<sycl::event> events(9);
    sycl::buffer<uint32_t> seed_buf = generate_seeds(q, N_connections, p.seed, events[0]);
    sycl::buffer<uint32_t, 2> events_to_buf = buffer_create_2D(q, std::vector<std::vector<uint32_t>>(p.Nt, std::vector<uint32_t>(N_connections, 0)), events[1]);
    sycl::buffer<uint32_t, 2> events_from_buf = buffer_create_2D(q, std::vector<std::vector<uint32_t>>(p.Nt, std::vector<uint32_t>(N_connections, 0)), events[2]);
    sycl::buffer<State_t, 2> community_state_buf = buffer_create_2D(q, std::vector<std::vector<State_t>>(p.Nt + 1, std::vector<State_t>(p.N_clusters, {0, 0, 0})), events[3]);
    sycl::buffer<SIR_State, 2> trajectory_buf = buffer_create_2D(q, std::vector<std::vector<SIR_State>>(p.Nt + 1, std::vector<SIR_State>(p.N_clusters * p.N_pop)), events[4]);
    sycl::buffer<float, 2> p_I_buf = buffer_create_2D(q, p_I_vec, events[5]);
    auto init_event = initialize_vertices(p.p_I0, p.p_R0, q, trajectory_buf, seed_buf, events);

    if (debug_flag)
    {
        end = std::chrono::steady_clock::now();
        log_file << "Inidividual buffers construction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }

    return enqueue_kernels(q, p, data, seed_buf, events_from_buf, events_to_buf, community_state_buf, trajectory_buf, p_I_buf, ccm,
                           output_dir, debug_flag, init_event);
}

void write_multiple_p_Is(const auto& p_I_vec, const std::string& output_dir)
{
    auto write_p_Is = [&, n=0](const auto& piv) mutable
    {
    std::ofstream p_I_f(output_dir + "/p_Is_" + std::to_string(n) + ".csv");
    std::for_each(piv.begin(), piv.end(), [&](const auto& pv)
    {
        linewrite(p_I_f, pv);
    });
    n++;
    };

    std::for_each(p_I_vec.begin(), p_I_vec.end(), write_p_Is);
}


void excite_simulation_read_to_files(const Sim_Param &p, sycl::queue &q, Sim_Data &d, const std::vector<uint32_t>& vcm, bool debug_flag)
{
    uint32_t N_clusters = p.N_clusters;
    uint32_t Nt = d.trajectory.get_range()[0] - 1;
    std::ofstream log_file;
    if (debug_flag)
        log_file.open(d.output_dir + "/Simulation_Debug.log");
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

    auto from_events = read_buffer(q, d.events_from_buf, d.event, log_file);
    auto to_events = read_buffer(q, d.events_to_buf, d.event, log_file);

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
}

void parallel_excite_simulate(const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir, uint32_t N_simulations, sycl::queue q, bool debug_flag)
{
    std::mt19937_64 rng(p.seed);

    std::vector<Sim_Param> sim_params(N_simulations, p);
    for (int i = 0; i < N_simulations; i++)
    {
        sim_params[i].seed = rng();
        sim_params[i].sim_idx = i;
    }
    std::vector<Sim_Data> simdata(N_simulations);

    std::transform(sim_params.begin(), sim_params.end(), simdata.begin(), [&](const Sim_Param &p)
                   { if(debug_flag)
                    std::cout << "Enqueuing Simulation " << p.sim_idx << std::endl;
                    return excite_simulate(p, vcm, edge_list, p_I_min, p_I_max, output_dir, debug_flag, q); });

    std::for_each(simdata.begin(), simdata.end(), [&](auto &d)
                  { if (debug_flag)
                    std::cout << "Reading Simulation " << d.sim_idx << std::endl;
                    excite_simulation_read_to_files(p, q, d, vcm, debug_flag); });
}
