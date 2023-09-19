
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <Sycl_Graph/Utils/json_settings.hpp>
#include <Sycl_Graph/Utils/path_config.hpp>
// #include <execution>
// #include <iomanip>
#include <CL/sycl.hpp>
#include <chrono>
auto make_iota(auto N)
{
    std::vector<uint32_t> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

auto make_vcms(auto N, auto N_pop, auto N_clusters)
{
    std::vector<std::vector<uint32_t>> vcms(N);
    std::vector<uint32_t> vcm(N_pop*N_clusters, 0);
    for(int i = 0; i < N_clusters; i++)
    {
        std::fill(vcm.begin(), vcm.begin() + N_pop, i);
    }
    std::fill(vcms.begin(), vcms.end(), vcm);
    return vcms;
}

auto make_edgelists(auto N, auto N_pop)
{
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_lists(N);
    std::generate(edge_lists.begin(), edge_lists.end(), [&N_pop]() { return complete_graph(N_pop);});
    return edge_lists;
}

Sim_Param create_sim_param(sycl::queue &q, uint32_t seed = 24)
{
    auto p = get_settings();
    auto device = q.get_device();
    p.wg_range = p.N_sims;
    p.compute_range = p.N_graphs * p.N_sims;
    p.seed = seed;
    p.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    p.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    return p;
}

int main()
{
    // parse argv which is N_sims, N_communities, N_pop

    std::chrono::high_resolution_clock::time_point t1, t2;

    sycl::queue q(sycl::cpu_selector_v);
    uint32_t seed = 283;
    auto p = create_sim_param(q);
    p.N_pop = 10;
    p.N_graphs = 1;
    p.N_communities = 2;
    p.N_sims = 1;
    p.wg_range = 1;
    p.compute_range = 1;

    auto vcms = make_vcms(p.N_graphs, p.N_pop, p.N_communities);
    auto edge_list = make_edgelists(p.N_graphs, p.N_pop);
    //max of ecms
    auto b = Sim_Buffers::make(q, p, edge_list, vcms, {});
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    run(q, p, b);

    return 0;
}
