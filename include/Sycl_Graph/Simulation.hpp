#ifndef SIR_SIMULATION_HPP
#define SIR_SIMULATION_HPP
#include <CL/sycl.hpp>
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
std::vector<std::vector<std::vector<T>>> remapTo3DVector(std::vector<T>&& input, size_t N0, size_t N1, size_t N2) {
    size_t M = N0 * N1 * N2;
    if (input.size() != M) {
        throw std::runtime_error("Input vector size does not match 3D dimensions.");
    }

    std::vector<std::vector<std::vector<T>>> output(N0, std::vector<std::vector<T>>(N1, std::vector<T>(N2)));

    size_t idx = 0;
    for (size_t i = 0; i < N0; ++i) {
        for (size_t j = 0; j < N1; ++j) {
            for (size_t k = 0; k < N2; ++k) {
                output[i][j][k] = input[idx++];
            }
        }
    }

    return output;
}



struct Simulation_Logger
{
    Simulation_Logger(const std::string& output_dir): profile_f(output_dir + "/profiling.log"), properties_f(output_dir + "/properties.log"), output_dir(output_dir) {}
    std::ofstream profile_f;
    std::ofstream properties_f;
    std::string output_dir;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    void write_community_state(cl::sycl::queue& q, sycl::buffer<State_t, 3>& community_state, sycl::event& event)
    {
        uint32_t N_sims = community_state.get_range()[0];
        uint32_t Nt = community_state.get_range()[1];
        uint32_t N_communities = community_state.get_range()[2];
        profile_f << "Community accumulation time: ";
        start = std::chrono::high_resolution_clock::now();
        event.wait();

        end = std::chrono::high_resolution_clock::now();
        profile_f << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

        profile_f << "File write time: ";
        start = std::chrono::high_resolution_clock::now();
        sycl::event event;
        auto community_state_vec = remapTo3DVector(std::forward<std::vector<State_t>>(read_buffer<SIR_State, 3>(community_state, q, event)), N_sims, Nt, N_communities);
        event.wait();
        std::for_each(community_state_vec.begin(), community_state_vec.end(), [&, n=0](const auto& vec) mutable {
            std::ofstream community_state_f(output_dir + "/community_state_" + std::to_string(i) + ".csv");
            std::for_each(vec.begin(), vec.end(), [&](const auto& inner_vec)
            {
                linewrite(community_state_f, inner_vec);
            });
            n++;
        });
        end = std::chrono::high_resolution_clock::now();
        profile_f << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    }

};

struct Simulator
{
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
    cl::sycl::buffer<State_t, 3> community_state;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::vector<uint32_t> ccm_weights;
    std::vector<sycl::event> events;
    Simulation_Logger logger;
    std::string output_dir;
    const uint32_t Nt, N_communities, N_pop, N_edges, N_wg, seed, N_connections, N_vertices, N_sims;
    void run();

private:
    void accumulate_community_state(std::vector<sycl::event> dep_events);
    sycl::nd_range<2> nd_range() const { return sycl::nd_range<2>(sycl::range<1>{N_sims}, sycl::range<1>{N_wg}); }
    std::ofstream profiling_f;
};

Simulator make_SIR_simulation(sycl::queue &q, const Sim_Param &p, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &vcm_init, const std::vector<float> &p_Is_init)
{
    auto ecm_init = ecm_from_vcm(edge_list, vcm_init);

    uint32_t N_connections = std::max_element(ecm_init.begin(), ecm_init.end())[0] + 1;

    // Initialize Common Buffers
    // get work group size
    auto device = q.get_device();
    auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();
    sycl::event rng_event;
    auto rngs = generate_rngs(q, sycl::range<2>(p.N_sims, N_wg), p.seed, rng_event);
    uint32_t N_vertices = vcm_init.size();
    auto trajectory = cl::sycl::buffer<SIR_State, 3>(sycl::range<3>(p.N_sims, N_vertices, p.Nt + 1));
    auto events_from = cl::sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, N_connections, p.Nt));
    auto events_to = cl::sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, N_connections, p.Nt));
    auto infections_from = cl::sycl::buffer<uint8_t, 2>(sycl::range<2>(p.N_sims, N_vertices));
    auto infections_to = cl::sycl::buffer<uint8_t, 2>(sycl::range<2>(p.N_sims, N_vertices));
    std::vector<sycl::event> alloc_events(6);
    auto p_Is = create_device_buffer<float, 3>(q, p_Is_init, sycl::range<3>(p.N_sims, p.Nt, N_connections), alloc_events[0]);

    std::vector<uint32_t> edge_from_init(edge_list.size());
    std::vector<uint32_t> edge_to_init(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), edge_from_init.begin(), [](auto &e)
                   { return e.first; });
    std::transform(edge_list.begin(), edge_list.end(), edge_to_init.begin(), [](auto &e)
                   { return e.second; });

    auto edge_from = create_device_buffer<uint32_t>(q, edge_from_init, alloc_events[1]);
    auto edge_to = create_device_buffer<uint32_t>(q, edge_to_init, alloc_events[2]);

    auto ecm = create_device_buffer<uint32_t>(q, ecm_init, alloc_events[3]);
    auto vcm = create_device_buffer<uint32_t>(q, vcm_init, alloc_events[4]);
    std::vector<State_t> community_state_init(p.N_communities * p.N_sims * (p.Nt + 1), {0, 0, 0});
    auto community_state = create_device_buffer<State_t, 3>(q, community_state_init, sycl::range<3>(p.N_sims, p.Nt + 1, p.N_communities), alloc_events[5]);

    auto ccm = complete_ccm(p.N_communities);
    auto ccm_weights = ccm_weights_from_ecm(ecm_init);

    return Simulator{q, rngs, trajectory, events_from, events_to,
                     infections_from, infections_to, p_Is,
                     edge_from, edge_to, ecm,
                     vcm, community_state, ccm, ccm_weights,
                     Simulation_Logger(p.output_dir), p.output_dir,
                     p.Nt, p.N_communities,
                     p.N_pop, edge_list.size(), N_wg, p.seed,
                     N_connections, p.N_pop*p.N_communities, p.N_sims};
}


#endif
