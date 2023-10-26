
#include <CL/sycl.hpp>
#include <SBM_Database/SBM_Database.hpp>
#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <SBM_Simulation/Graph/Community_Mappings.hpp>
#include <SBM_Simulation/Graph/Graph.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <Sycl_Buffer_Routines/Buffer_Routines.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>
#include <chrono>

auto nested_vec_max(const std::vector<std::vector<uint32_t>> &vec)
{
    auto max = std::max_element(vec.begin(), vec.end(), [](const auto &a, const auto &b)
                                { return *std::max_element(a.begin(), a.end()) < *std::max_element(b.begin(), b.end()); });
    return *std::max_element(max->begin(), max->end());
}

Sim_Param create_sim_param(uint32_t N_communities)
{
    Sim_Param p;
    p.N_pop = 10;
    p.N_graphs = 2;
    p.N_communities = std::vector<uint32_t>(p.N_graphs, N_communities);
    p.p_in = 0.5f;
    p.p_out = 0.1f;
    p.N_sims = 2;
    p.Nt = 56;
    p.Nt_alloc = 20;
    p.seed = 234;
    p.p_I_min = 0.1f;
    p.p_I_max = 0.2f;
    p.p_out_idx = 0;
    p.p_R = 0.1f;
    p.p_I0 = 0.1f;
    p.p_R0 = 0.0f;
    return p;
}
int main()
{
    using namespace SBM_Database;

    // project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;

    sycl::queue q(sycl::cpu_selector_v);
    uint32_t seed = 283;
    auto N_communities = 2;
    auto p = create_sim_param(N_communities);

    std::vector<uint32_t> p_out(10, 0);

    auto [edge_lists, vertex_list] = generate_N_SBM_graphs(p.N_pop, N_communities, p.p_in, p.p_out, p.seed, p.N_graphs);

    return 0;
}
