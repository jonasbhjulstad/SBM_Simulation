
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Multiple_Simulation.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <Sycl_Graph/Utils/math.hpp>

// #include <execution>
// #include <iomanip>
#include <CL/sycl.hpp>
#include <chrono>

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


auto generate_graphs(sycl::queue& q, const std::string& output_dir)
{
    auto N_p_out = 5;
    Multiple_Sim_Param_t p(output_dir + "Sim_Param.json");
    p.p_out = make_linspace(0.0f, 1.0f, 0.1f);
    Dataframe_t<std::pair<uint32_t, uint32_t>, 3> edge_lists(N_p_out);
    Dataframe_t<uint32_t, 3> vcms(N_p_out);

    for(int i = 0; i < N_p_out; i++)
    {
        auto [edge_list, vertex_list, ecm, vcm] = generate_N_SBM_graphs_flat(p.N_pop, p.N_communities, p.p_in, p.p_out[i], p.seed, p.N_graphs);
        edge_lists[i] = edge_list;
        vcms[i] = vcm;
    }
    return std::make_tuple(p, edge_lists, vcms);
}

int main()
{
    //project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;

    sycl::queue q(sycl::cpu_selector_v);
    uint32_t seed = 283;

    auto [p, edge_lists, vcms] = generate_graphs(q, output_dir);
    multiple_sim_param_run(q, p, edge_lists, vcms);
    auto sim_idx = 0;
    for(int i = 0; i < p.p_out.size(); i++)
    {
        auto p_out_str = float_to_decimal_string(p.p_out[i]);
        auto graph_dir = p.output_dir + p_out_str + "/Graph_";
        auto graph_edgelists = edge_lists[i].slice(sim_idx, sim_idx + p.N_sims);
        for(int j = 0; j < graph_edgelists.size(); j++)
        {
            write_edgelist(graph_dir + std::to_string(j) + "/edgelist.csv", graph_edgelists[j]);
        }
        sim_idx+= p.N_sims;
    }

    return 0;
}
