
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
// #include <execution>
// #include <iomanip>
#include <CL/sycl.hpp>
#include <chrono>

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

auto nested_vec_max(const std::vector<std::vector<uint32_t>>& vec)
{
    auto max = std::max_element(vec.begin(), vec.end(), [](const auto& a, const auto& b)
    {
        return *std::max_element(a.begin(), a.end()) < *std::max_element(b.begin(), b.end());
    });
    return *std::max_element(max->begin(), max->end());
}

int main()
{
    // parse argv which is N_sims, N_communities, N_pop

    std::chrono::high_resolution_clock::time_point t1, t2;

    sycl::queue q(sycl::cpu_selector_v);
    uint32_t seed = 283;
    auto p = create_sim_param(q);
    // auto p = ps[0];

    t1 = std::chrono::high_resolution_clock::now();

    auto [edge_list, vertex_list, ecm, vcm] = generate_N_SBM_graphs_flat(p.N_pop, p.N_communities, p.p_in, p.p_out, p.seed, p.N_graphs);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Generate graphs: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;
    std::mt19937 rng(p.seed);
    std::uniform_int_distribution<uint32_t> dist_v(0, 0);
    for(auto& v: vcm)
    {
       std::generate(v.begin(), v.end(), [&dist_v, &rng]() { return dist_v(rng); });
    }

    //max of ecms
    auto b = Sim_Buffers::make(q, p, edge_list, vcm, {});
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Make buffers: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    run(q, p, b);
    std::for_each(edge_list.begin(), edge_list.end(), [p,i=0](const auto& edges) mutable
    {
        write_edgelist(p.output_dir + "Graph_" + std::to_string(i) + "/edgelist.csv", edges);
    });

    return 0;
}
