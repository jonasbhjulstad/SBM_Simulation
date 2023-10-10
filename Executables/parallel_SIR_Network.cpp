
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <CL/sycl.hpp>
#include <chrono>
#include <Sycl_Graph/Database/Simulation_Tables.hpp>
Sim_Param create_sim_param(sycl::queue &q, const std::string& fname, uint32_t seed = 24)
{
    auto p = Sim_Param(fname);
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
    //project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;

    sycl::queue q(sycl::cpu_selector_v);
    uint32_t seed = 283;
    auto p = create_sim_param(q, output_dir + "Sim_Param.json");

    auto con = pqxx::connection("dbname=postgres user=postgres");

    construct_simulation_tables(con, 1, p.N_graphs, p.N_sims, p.Nt+1);

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
    });

    drop_simulation_tables(con);

    return 0;
}
