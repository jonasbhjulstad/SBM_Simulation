#ifndef SIR_SIMULATION_HPP
#define SIR_SIMULATION_HPP
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <chrono>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
struct Common_Buffers;

struct Common_Buffers
{
    Common_Buffers(sycl::queue &q, const std::vector<uint32_t> &edge_from_init, const std::vector<uint32_t> &edge_to_init, const std::vector<uint32_t> &ecm_init, const std::vector<uint32_t> &vcm_init, uint32_t N_communities, uint32_t N_connections, uint32_t Nt, uint32_t seed);
    Common_Buffers(const auto p_edge_from, const auto p_edge_to, const auto p_ecm, const auto p_vcm, const auto &ccm, const auto &ccm_weights, const std::vector<sycl::event> &events);
    Common_Buffers(const Common_Buffers &other);

    std::string get_sizes();

    uint32_t N_connections() const;
    std::vector<sycl::event> events = std::vector<sycl::event>(4);
    std::shared_ptr<sycl::buffer<uint32_t>> edge_from;
    std::shared_ptr<sycl::buffer<uint32_t>> edge_to;
    std::shared_ptr<sycl::buffer<uint32_t>> ecm;
    std::shared_ptr<sycl::buffer<uint32_t>> vcm;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::vector<uint32_t> ccm_weights;
};

template <typename T>
std::vector<std::vector<std::vector<T>>> vector_remap(std::vector<T> &&input, size_t N0, size_t N1, size_t N2)
{
    size_t M = N0 * N1 * N2;
    if (input.size() != M)
    {
        throw std::runtime_error("Input vector size does not match 3D dimensions.");
    }

    std::vector<std::vector<std::vector<T>>> output(N0, std::vector<std::vector<T>>(N1, std::vector<T>(N2)));

    size_t idx = 0;
    for (size_t i = 0; i < N0; ++i)
    {
        for (size_t j = 0; j < N1; ++j)
        {
            for (size_t k = 0; k < N2; ++k)
            {
                output[i][j][k] = input[idx++];
            }
        }
    }

    return output;
}

struct Simulation_Logger
{
    Simulation_Logger(const std::string &output_dir) : profile_f(output_dir + "/profiling.log"), properties_f(output_dir + "/properties.log"), output_dir(output_dir) {}
    std::ofstream profile_f;
    std::ofstream properties_f;
    std::string output_dir;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    void log_time()
    {
        profile_f << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    }

    void log_start()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    void log_end()
    {
        end = std::chrono::high_resolution_clock::now();
        log_time();
    }

    void profile_events(std::vector<sycl::event> &events)
    {
        start = std::chrono::high_resolution_clock::now();
        for (auto &event : events)
        {
            event.wait();
        }
        end = std::chrono::high_resolution_clock::now();
        log_time();
    }
};

struct Simulator
{
    Simulator(sycl::queue &q,
              cl::sycl::buffer<Static_RNG::default_rng, 2> &rngs,
              cl::sycl::buffer<SIR_State, 3> &trajectory,
              cl::sycl::buffer<uint32_t, 3> &events_from,
              cl::sycl::buffer<uint32_t, 3> &events_to,
              cl::sycl::buffer<uint8_t, 2> &infections_from,
              cl::sycl::buffer<uint8_t, 2> &infections_to,
              cl::sycl::buffer<float, 3> &p_Is,
              cl::sycl::buffer<uint32_t> &edge_from,
              cl::sycl::buffer<uint32_t> &edge_to,
              cl::sycl::buffer<uint32_t> &ecm,
              cl::sycl::buffer<uint32_t> &vcm,
              const std::vector<uint32_t> &vcm_vec,
              cl::sycl::buffer<State_t, 3> &community_state,
              const std::vector<std::pair<uint32_t, uint32_t>> &ccm,
              const std::vector<uint32_t> &ccm_weights,
              Simulation_Logger logger,
              const std::string &output_dir,
              const uint32_t Nt,
              const uint32_t N_communities,
              const uint32_t N_pop,
              const uint32_t N_edges,
              const uint32_t N_threads,
              const uint32_t seed,
              const uint32_t N_connections,
              const uint32_t N_vertices,
              const uint32_t N_sims,
              const float p_I0, const float p_R0, const float p_R,
              std::vector<sycl::event> alloc_events);
    sycl::queue &q;
    cl::sycl::buffer<Static_RNG::default_rng, 2> rngs;
    cl::sycl::buffer<SIR_State, 3> trajectory;
    cl::sycl::buffer<uint32_t, 3> events_from;
    cl::sycl::buffer<uint32_t, 3> events_to;
    cl::sycl::buffer<uint8_t, 2> infections_from;
    cl::sycl::buffer<uint8_t, 2> infections_to;
    cl::sycl::buffer<float, 3> p_Is;

    cl::sycl::buffer<uint32_t> edge_from;
    cl::sycl::buffer<uint32_t> edge_to;
    cl::sycl::buffer<uint32_t> ecm;
    cl::sycl::buffer<uint32_t> vcm;
    const std::vector<uint32_t> vcm_vec;
    cl::sycl::buffer<State_t, 3> community_state;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::vector<uint32_t> ccm_weights;
    // std::vector<sycl::event> events;
    Simulation_Logger logger;
    std::string output_dir;
    const uint32_t Nt, N_communities, N_pop, N_edges, N_threads, seed, N_connections, N_vertices, N_sims;
    const float p_I0, p_R0, p_R;
    std::vector<sycl::event> alloc_events;
    const uint32_t max_infection_samples = 1000;
    std::vector<sycl::event> enqueue();
    void run();
    sycl::event initialize_vertices();

    std::vector<sycl::event> recover(uint32_t t, std::vector<sycl::event> &dep_event);
    std::vector<sycl::event> infect(uint32_t t, std::vector<sycl::event> &dep_event);
    void write_to_files(std::vector<sycl::event> &dep_events);

    uint32_t N_vertex_per_thread() const;
    uint32_t N_edge_per_thread() const;

private:
    std::vector<std::vector<std::vector<uint32_t>>> sample_from_connection_events(const std::vector<std::vector<std::vector<State_t>>> &community_state, const std::vector<std::vector<std::vector<uint32_t>>> &from_events,
                                                                                  const std::vector<std::vector<std::vector<uint32_t>>> &to_events);

    std::vector<std::vector<std::vector<State_t>>> accumulate_community_state(std::vector<sycl::event> &events, sycl::event &res_event);
    uint32_t get_work_group_size() const;
    sycl::nd_range<1> get_nd_range() const;
    std::ofstream profiling_f;
};

Simulator make_SIR_simulation(sycl::queue &q, const Sim_Param &p, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &vcm_init, const std::vector<float> &p_Is_init);

Simulator make_SIR_simulation(sycl::queue &q, const Sim_Param &p, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &vcm_init, const float p_I_min, const float p_I_max);

#endif
