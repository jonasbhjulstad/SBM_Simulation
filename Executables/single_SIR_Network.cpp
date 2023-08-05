#define TBB_DEBUG 1
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/SIR_Dynamics.hpp>
#include <Sycl_Graph/SIR_Infection_Sampling.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>
#include <algorithm>
#include <execution>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>


template <typename T>
auto merge_vectors(const std::vector<std::vector<T>> &vectors)
{
    std::vector<T> merged;
    uint32_t size = 0;
    for(int i = 0; i < vectors.size(); i++)
    {
        size += vectors[i].size();
    }
    merged.reserve(size);
    for (auto &v : vectors)
    {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}

int main()
{
    std::string output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Graph_" + std::to_string(0) + "/";

    uint32_t N_clusters = 2;
    uint32_t N_pop = 100;
    uint32_t N_pop_tot = N_pop * N_clusters;
    float p_in = 1.0f;
    float p_out = 1.0f;
    uint32_t N_sims = 2;
    uint32_t Ng = 1;
    // sycl::queue q(sycl::gpu_selector_v);
    sycl::queue q(sycl::cpu_selector_v);
    // get work group size
    auto device = q.get_device();
    auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();

    uint32_t Nt = 10;
    uint32_t seed = 100;
    uint32_t N_threads = 10;


    auto [edge_lists, node_lists] = generate_planted_SBM_edges(N_pop, N_clusters, p_in, p_out, seed);

    auto edge_list = merge_vectors(edge_lists);

    auto ccm = complete_ccm(N_clusters);
    auto ecm = create_ecm(edge_lists);
    auto vcm = create_vcm(node_lists);
    std::vector<uint32_t> ccm_weights(edge_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), ccm_weights.begin(), [&ccm](auto &e)
                   { return e.size(); });

    uint32_t N_connections = edge_lists.size();

    float p_I_min = 1e-3f;
    float p_I_max = 1e-2f;
    float p_R = 1e-1f;

    auto p_I0 = 0.1f;
    auto p_R0 = 0.0f;
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

    // recovery
    uint32_t t = 0;

    sycl::event rec_event;
    sycl::event inf_event;

    for (int t = 0; t < Nt; t++)
    {
        auto rec_event = recover(q, t, inf_event, p_R, seed_buf, trajectory, vcm_buf);

        inf_event = infect(q, ecm_buf, p_I_buf, seed_buf, event_from_buf, event_to_buf, trajectory, edge_from_buf, edge_to_buf, N_wg, t, N_connections, rec_event);
    }
    inf_event.wait();
    auto vertex_state = read_buffer(q, trajectory, inf_event);

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

    auto from_events = read_buffer(q, event_from_buf, inf_event);
    auto to_events = read_buffer(q, event_to_buf, inf_event);

    auto connection_infections = sample_infections(community_state, from_events, to_events, ccm, ccm_weights, seed);

    std::vector<std::vector<uint32_t>> connection_events(from_events.size(), std::vector<uint32_t>(from_events[0].size()));

    for (int t = 0; t < from_events.size(); t++)
    {
        for (int i = 0; i < from_events.size(); i++)
        {
            connection_events[t][2 * i] = from_events[t][i];
            connection_events[t][2 * i + 1] = to_events[t][i];
        }
    }

    std::filesystem::create_directory(
        std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
    std::filesystem::create_directories(output_dir);
    uint32_t sim_idx = 0;

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
    std::ofstream p_I_f(output_dir + "p_Is_" + std::to_string(sim_idx) + ".csv");
    std::for_each(p_I_vec.begin(), p_I_vec.end(),
                  [&](auto &p_I_i)
                  { linewrite(p_I_f, p_I_i); });

    return 0;
}
