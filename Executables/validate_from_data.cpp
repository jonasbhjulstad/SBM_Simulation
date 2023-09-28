
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <Sycl_Graph/Utils/json_settings.hpp>
#include <CL/sycl.hpp>
#include <chrono>
#include <filesystem>


auto get_p_dirs()
{
    auto cwd = std::filesystem::current_path();
    auto data_dir = std::string(cwd) + "/SIR_sim/";
    //create data dir if it doesn't exist
    if (!std::filesystem::exists(data_dir))
    {
        std::filesystem::create_directories(data_dir);
    }
    std::vector<std::string> dir_list;
    for (auto &p : std::filesystem::directory_iterator(data_dir))
    {
        if (p.is_directory())
        {
            dir_list.push_back(p.path());
        }
    }
    return dir_list;
}

auto get_graph_dirs(const std::string& dir)
{
    std::vector<std::string> dir_list;
    for (auto &p : std::filesystem::directory_iterator(dir))
    {
        //if is dir and starts with Graph_
        if (p.is_directory() && p.path().filename().string().substr(0, 6) == "Graph_")
        {
            dir_list.push_back(p.path());
        }
    }
    return dir_list;
}

void compute_for_p(const std::string& p_dir)
{
    auto p = Sim_Param(p_dir + "/Sim_Param.json");
    p.output_dir = p_dir + "/Results/";
    p.simulation_subdir = "Detected";
    sycl::queue q(sycl::cpu_selector_v);

    auto graph_dirs = get_graph_dirs(p_dir);
    assert(graph_dirs.size() == p.N_graphs);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_lists(p.N_graphs);
    std::transform(graph_dirs.begin(), graph_dirs.end(), edge_lists.begin(), [](const auto& dir)
    {
        return read_edgelist(dir + "/edgelist.csv");
    });
    std::vector<std::vector<uint32_t>> vcms(p.N_graphs);
    std::transform(graph_dirs.begin(), graph_dirs.end(), vcms.begin(), [p](const auto& dir)
    {
        return read_vec(dir + "/vcm.csv", p.N_pop*p.N_communities);
    });
    auto b = Sim_Buffers::make(q, p, edge_lists, vcms, {});
    run(q, p, b);

}


int main()
{
    auto p_dirs = get_p_dirs();
    std::for_each(p_dirs.begin(), p_dirs.end(), [](auto p_dir)
    {
        compute_for_p(p_dir);
    });

    return 0;
}
