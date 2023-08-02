#define TBB_DEBUG 1
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/SBM_Generation.hpp>
#include <Sycl_Graph/SBM_write.hpp>
#include <Sycl_Graph/SIR_SBM_Network.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

using namespace Sycl_Graph::SBM;

int main()
{
    uint32_t N_clusters = 2;
    uint32_t N_pop = 100;
    float p_in = 1.0f;
    float p_out = 1.0f;
    uint32_t N_sims = 2;
    uint32_t Ng = 1;
    // sycl::queue q(sycl::gpu_selector_v);
    sycl::queue q(sycl::gpu_selector_v);
    //get work group size
    auto device = q.get_device();
    auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();

    uint32_t Nt = 70;
    uint32_t seed = 47;
    uint32_t N_threads = 10;

    auto G =
        create_planted_SBM(N_pop, N_clusters, p_in, p_out, N_threads, seed);

    float p_I_min = 1e-5f;
    float p_I_max = 1e-3f;
    float p_R = 1e-1f;

    std::string output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Graph_" + std::to_string(0) + "/";

    std::vector<std::vector<float>> p_I_vec = generate_p_Is(G.N_connections, p_I_min, p_I_max, Nt, seed);
    std::filesystem::create_directory(
        std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");

    SIR_SBM_Param_t param;
    param.p_R = p_R;
    param.p_I = p_I_vec;

    param.p_I0 = 0.1f;
    param.p_R0 = 0.0f;


    simulate_to_file(G, param, q, output_dir, 0, seed, N_wg);

    return 0;
}
