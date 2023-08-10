
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

Common_Buffers::Common_Buffers(sycl::queue &q,
                               const std::vector<uint32_t> &edge_from_init,
                               const std::vector<uint32_t> &edge_to_init,
                               const std::vector<uint32_t> &ecm_init,
                               const std::vector<uint32_t> &vcm_init,
                               uint32_t N_clusters, uint32_t N_connections,
                               uint32_t Nt, uint32_t seed) : edge_from{shared_buffer_create_1D(q, edge_from_init, events[0])},
                                                             edge_to{shared_buffer_create_1D(q, edge_to_init, events[1])},
                                                             ecm{shared_buffer_create_1D(q, ecm_init, events[2])},
                                                             vcm{shared_buffer_create_1D(q, vcm_init, events[3])},
                                                             ccm{complete_ccm(N_clusters)},
                                                             ccm_weights{ccm_weights_from_ecm(ecm_init)}
{
}
Common_Buffers::Common_Buffers(const auto p_edge_from, const auto p_edge_to, const auto p_ecm, const auto p_vcm, const auto &ccm, const auto &ccm_weights, const std::vector<sycl::event> &events)
    : edge_from{p_edge_from}, edge_to{p_edge_to}, ecm{p_ecm}, vcm{p_vcm}, ccm{ccm}, ccm_weights{ccm_weights}, events(events)
{
}
Common_Buffers::Common_Buffers(const Common_Buffers &other) : Common_Buffers(other.edge_from, other.edge_to, other.ecm, other.vcm, other.ccm, other.ccm_weights, other.events) {}

std::string Common_Buffers::get_sizes()
{
    std::stringstream ss;
    ss << "edge_from: " << edge_from->size() << std::endl;
    ss << "edge_to: " << edge_to->size() << std::endl;
    ss << "ecm: " << ecm->size() << std::endl;
    ss << "vcm: " << vcm->size() << std::endl;
    ss << "ccm: " << ccm.size() << std::endl;
    ss << "ccm_weights: " << ccm_weights.size() << std::endl;
    return ss.str();
}

uint32_t Common_Buffers::N_connections() const
{
    return ccm.size();
}

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

Common_Buffers allocate_common_buffers(sycl::queue &q, const auto &edge_list, const auto &vcm, uint32_t Nt, uint32_t N_clusters, uint32_t seed)
{
    auto ecm = ecm_from_vcm(edge_list, vcm);
    uint32_t N_connections = std::max_element(ecm.begin(), ecm.end())[0] + 1;

    // Initialize Common Buffers
    std::vector<uint32_t> edge_from_init(edge_list.size());
    std::vector<uint32_t> edge_to_init(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), edge_from_init.begin(), [](auto &e)
                   { return e.first; });
    std::transform(edge_list.begin(), edge_list.end(), edge_to_init.begin(), [](auto &e)
                   { return e.second; });
    return Common_Buffers(q, edge_from_init, edge_to_init, ecm, vcm, N_clusters, N_connections, Nt, seed);
}

template <typename T>
std::vector<T> read_pointer(const std::shared_ptr<T> &p, uint32_t N)
{
    std::vector<T> result(N);
    std::copy(p.get(), p.get() + N, result.begin());
    return result;
}

template <typename T>
std::vector<std::vector<T>> read_pointer(const std::shared_ptr<T> &p, uint32_t rows, uint32_t cols)
{
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols));
    for (int i = 0; i < rows; i++)
    {
        std::copy(p.get() + i * cols, p.get() + (i + 1) * cols, result[i].begin());
    }
    return result;
}

struct Individual_Buffer_Data
{
    std::vector<std::vector<SIR_State>> vertex_state;
    std::vector<std::vector<uint32_t>> from_events;
    std::vector<std::vector<uint32_t>> to_events;
};

struct Individual_Buffers
{
    Individual_Buffers(sycl::queue &q, const std::vector<std::vector<float>> &p_Is, uint32_t N_connections, uint32_t Nt, uint32_t N_clusters, uint32_t N_pop, uint32_t N_edges, uint32_t N_wg, uint32_t seed)

        : events(2), seeds{generate_seeds(q, N_wg, seed, events[0])}, events_to{sycl::buffer<uint32_t, 2>(sycl::range<2>(Nt, N_connections))},
          trajectory{sycl::buffer<SIR_State, 2>(sycl::range<2>(Nt + 1, N_pop * N_clusters))},
          events_from{sycl::buffer<uint32_t, 2>(sycl::range<2>(Nt, N_connections))},
          infection_indices{sycl::range<1>(N_edges)},
          p_Is{buffer_create_2D(q, p_Is, events[1])}
    {
    }
    // default copy constructor
    Individual_Buffers(const Individual_Buffers &) = default;
    std::vector<sycl::event> events = std::vector<sycl::event>(2);

    sycl::buffer<uint32_t> seeds;
    sycl::buffer<SIR_State, 2> trajectory;
    sycl::buffer<uint32_t, 2> events_from;
    sycl::buffer<uint32_t, 2> events_to;
    sycl::buffer<uint32_t> infection_indices;
    sycl::buffer<float, 2> p_Is;

    sycl::event initialize_trajectory(sycl::queue &q, float p_I0, float p_R0, std::vector<sycl::event> dep_event)
    {
        return initialize_vertices(p_I0, p_R0, q, trajectory, seeds, dep_event);
    }

    static std::vector<Individual_Buffers> make(sycl::queue &q, const std::vector<std::vector<std::vector<float>>> &p_Is_vec, uint32_t N_connections, uint32_t Nt, uint32_t N_clusters, uint32_t N_pop, uint32_t N_edges, uint32_t N_wg, const std::vector<uint32_t> &init_seeds)
    {
        std::vector<Individual_Buffers> result;
        result.reserve(init_seeds.size());
        std::transform(p_Is_vec.begin(), p_Is_vec.end(), init_seeds.begin(), std::back_inserter(result), [&](const auto &p_Is, uint32_t init_seed)
                       { return Individual_Buffers(q, p_Is, N_connections, Nt, N_clusters, N_pop, N_edges, N_wg, init_seed); });
        return result;
    }

    static std::vector<Individual_Buffers> make(sycl::queue &q, const std::vector<std::vector<float>> &p_Is, uint32_t N_connections, uint32_t Nt, uint32_t N_clusters, uint32_t N_pop, uint32_t N_edges, uint32_t N_wg, const std::vector<uint32_t> &init_seeds)
    {
        std::vector<Individual_Buffers> result;
        result.reserve(init_seeds.size());
        std::transform(init_seeds.begin(), init_seeds.end(), std::back_inserter(result), [&](uint32_t init_seed)
                       { return Individual_Buffers(q, p_Is, N_connections, Nt, N_clusters, N_pop, N_edges, N_wg, init_seed); });
        return result;
    }
    Individual_Buffer_Data read_buffers(sycl::queue &q, auto dep_event)
    {

        std::vector<sycl::event> read_events(3);
        auto p_vertex_state = read_buffer(q, this->trajectory, dep_event, read_events[0]);
        auto p_from_events = read_buffer(q, this->events_from, dep_event, read_events[1]);
        auto p_to_events = read_buffer(q, this->events_to, dep_event, read_events[2]);

        std::for_each(read_events.begin(), read_events.end(), [](auto &e)
                      { e.wait(); });
        return {read_pointer(p_vertex_state, this->trajectory.get_range()[0], this->trajectory.get_range()[1]),
                read_pointer(p_from_events, this->events_from.get_range()[0], this->events_from.get_range()[1]),
                read_pointer(p_to_events, this->events_to.get_range()[0], this->events_to.get_range()[1])};
    };
};

sycl::event enqueue_timeseries(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, Individual_Buffers &ib, sycl::event dep_event)
{
    sycl::event inf_event, rec_event;
    std::chrono::steady_clock::time_point begin, end;
    std::cout << "Enqueueing timeseries..." << std::endl;
    for (int t = 0; t < p.Nt; t++)
    {
        std::cout << "Enqueueing timestep " << t << " of " << p.Nt << "..." << std::endl;
        std::cout << "Recovery [";
        begin = std::chrono::steady_clock::now();
        rec_event = recover(q, t, dep_event, p.p_R, ib.seeds, ib.trajectory);
        end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms]" << std::endl;
        std::cout << "Infection [" << std::endl;
        begin = std::chrono::steady_clock::now();
        inf_event = infect(q, cb.ecm, ib.p_Is, ib.seeds, ib.events_from, ib.events_to, ib.trajectory, cb.edge_from, cb.edge_to, ib.infection_indices, t, cb.N_connections(), rec_event);
        end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms]" << std::endl;
    }
    return inf_event;
}

std::vector<std::vector<State_t>> to_community_state(sycl::queue &q, const std::vector<std::vector<SIR_State>> &vertex_state, const std::vector<uint32_t> &vcm, sycl::event dep_event)
{
    uint32_t Nt = vertex_state.size() - 1;
    uint32_t N_clusters = std::max_element(vcm.begin(), vcm.end())[0] + 1;
    std::vector<std::vector<State_t>> community_state(Nt + 1, std::vector<State_t>(N_clusters, {0, 0, 0}));
    std::transform(std::execution::par_unseq, vertex_state.begin(), vertex_state.end(), community_state.begin(), [=](auto v_state)
                   {

            std::vector<State_t> state(N_clusters, {0, 0, 0});
            for(int i = 0; i < v_state.size(); i++)
            {
                auto community_idx = vcm[i];
                state[community_idx][v_state[i]]++;
            }
            return state; });
    return community_state;
}


void write_to_file(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, Individual_Buffers &ib, const std::vector<uint32_t> &vcm, const std::string &output_dir, sycl::event dep_event, uint32_t sim_idx, uint32_t seed)
{

    std::vector<sycl::event> read_events(3);
    auto buffer_data = ib.read_buffers(q, dep_event);
    auto community_state = to_community_state(q, buffer_data.vertex_state, vcm, dep_event);

    auto connection_infections = sample_infections(community_state, buffer_data.from_events, buffer_data.to_events, cb.ccm, cb.ccm_weights, seed, p.max_infection_samples);

    auto connection_events = column_zip_2D(buffer_data.from_events, buffer_data.to_events);
    std::filesystem::create_directories(output_dir);

    std::ofstream community_traj_f(output_dir + "community_trajectory_" +
                                   std::to_string(sim_idx) + ".csv");
    std::ofstream connection_events_f(output_dir + "connection_events_" +
                                      std::to_string(sim_idx) + ".csv");

    std::ofstream connection_infections_f(output_dir + "connection_infections_" +
                                          std::to_string(sim_idx) + ".csv");

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

sycl::event single_enqueue(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, Individual_Buffers &ib, const std::string output_dir)
{
    auto dep_event = ib.initialize_trajectory(q, p.p_I0, p.p_R0, ib.events);
    return enqueue_timeseries(q, p, cb, ib, dep_event);
}
uint32_t max_work_group_size(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    return max_wg_size;
}

void simulate(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<std::vector<float>> &p_Is, const std::string output_dir, uint32_t N_simulations, uint32_t seed)
{
    auto N_edges = edge_list.size();
    auto N_wg = max_work_group_size(q);
    std::vector<sycl::event> events(N_simulations);

    auto seeds = generate_seeds(N_simulations, seed);
    std::vector<Individual_Buffers> ibs;
    ibs.reserve(N_simulations);

    std::generate_n(std::back_inserter(ibs), N_simulations, [&, n = 0]() mutable
                    { return Individual_Buffers(q, p_Is, cb.N_connections(), p.Nt, p.N_clusters, p.N_pop, N_edges, N_wg, seed); });
    std::transform(ibs.begin(), ibs.end(), events.begin(), [&](auto &ib)
                   { return single_enqueue(q, p, cb, ib, output_dir); });

    for (int i = 0; i < N_simulations; i++)
    {
        write_to_file(q, p, cb, ibs[i], vcm, output_dir, events[i], i, seeds[i]);
    }
}

void excite_simulate(sycl::queue &q, const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir, uint32_t N_simulations, uint32_t seed)
{
    std::chrono::steady_clock::time_point begin, end;

    begin = std::chrono::steady_clock::now();
    auto N_edges = edge_list.size();
    auto N_wg = max_work_group_size(q);
    auto seeds = generate_seeds(N_simulations, seed);
    std::vector<sycl::event> events(N_simulations);
    auto cb = allocate_common_buffers(q, edge_list, vcm, p.Nt, p.N_clusters, seed);
    uint32_t N_connections = cb.N_connections();
    auto p_Is_vec = generate_floats(N_simulations, p.Nt, N_connections, p_I_min, p_I_max, seed);
    auto ibs = Individual_Buffers::make(q, p_Is_vec, N_connections, p.Nt, p.N_clusters, p.N_pop, N_edges, N_wg, seeds);
    end = std::chrono::steady_clock::now();
    std::cout << "Buffer construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    begin = std::chrono::steady_clock::now();
    std::transform(ibs.begin(), ibs.end(), events.begin(), [&](auto &ib)
                   { return single_enqueue(q, p, cb, ib, output_dir); });
    end = std::chrono::steady_clock::now();
    std::cout << "Enqueue: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N_simulations; i++)
    {
        write_to_file(q, p, cb, ibs[i], vcm, output_dir, events[i], i, seeds[i]);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Write to file: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
}
