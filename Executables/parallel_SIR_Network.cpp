
#include <CL/sycl.hpp>
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

int main()
{
    using namespace SBM_Database;

    // project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;
    auto N_communities = 2;
    auto N_pop = 10;
    auto N_graphs = 2;
    auto N_communities = std::vector<uint32_t>(p.N_graphs, N_communities);
    auto p_in = 0.5f;
    auto p_out = 0.1f;
    auto N_sims = 2;
    auto Nt = 56;
    auto Nt_alloc = 20;
    auto seed = 234;
    auto p_I_min = 0.1f;
    auto p_I_max = 0.2f;
    auto p_out_idx = 0;
    auto p_R = 0.1f;
    auto p_I0 = 0.1f;
    auto p_R0 = 0.0f;


    sycl::queue q(sycl::cpu_selector_v);
    uint32_t seed = 283;
    auto N_communities = 2;
    auto p = create_simulation_parameters(N_pop,N_graphs,N_communities, p_in, p_out, N_sims, Nt, Nt_alloc, seed, p_I_min, p_I_max, p_R, p_I0, p_R0);


    // auto index_names = merge_arrays(constant_indices, iterable_index_names);

    t1 = std::chrono::high_resolution_clock::now();

    auto [edge_lists, vertex_list] = generate_N_SBM_graphs(p.N_pop, N_communities, p.p_in, p.p_out, p.seed, p.N_graphs);

    auto vcms = std::vector<std::vector<uint32_t>>(p.N_graphs, create_vcm(vertex_list[0]));

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Generate graphs: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    std::mt19937 rng(p.seed);
    // std::uniform_int_distribution<uint32_t> dist_v(0, 0);
    // for(auto& v: vcms)
    // {
    //    std::generate(v.begin(), v.end(), [&dist_v, &rng]() { return dist_v(rng); });
    // }u
    Simulation_t sim(q, p, edge_lists, vcms);
    Buffer_Routines::validate_buffer_elements<uint32_t, 2>(q, *sim.b.vcm, [](auto v){ return v < 2; });
    sim.run();
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();

    return 0;
}
