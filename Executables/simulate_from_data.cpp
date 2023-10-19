
#include <SBM_Simulation/Graph/Graph.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>
#include <CL/sycl.hpp>
#include <chrono>
#include <filesystem>


auto get_p_dirs(const std::string& data_dir)
{
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

auto get_basename_p_dirs(const std::string& data_dir)
{
    auto p_dirs = get_p_dirs(data_dir);
    //return /basename/
    std::transform(p_dirs.begin(), p_dirs.end(), p_dirs.begin(), [](const auto& dir)
    {
        return dir.substr(dir.find_last_of("/\\") + 1);
    });
    return p_dirs;
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

void compute_for_p(const std::string& root_data_dir, const std::string& relative_p_dir)
{
    auto absolute_p_dir = root_data_dir + relative_p_dir;
    //check that file exists
    if (!std::filesystem::exists(absolute_p_dir + "/Sim_Param.json"))
    {
        throw std::runtime_error("Sim_Param.json not found in " + absolute_p_dir);
    }

    auto p = Sim_Param(absolute_p_dir + "/Sim_Param.json");
    p.output_dir = root_data_dir + "/Excitation/" + relative_p_dir;
    sycl::queue q(sycl::cpu_selector_v);
    std::string device_name = q.get_device().get_info<sycl::info::device::name>();

    auto graph_dirs = get_graph_dirs(absolute_p_dir);
    assert(graph_dirs.size() == p.N_graphs);
    std::vector<std::vector<Edge_t>> edge_lists(p.N_graphs);
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
std::string get_data_dir(int argc, char** argv)
{
    std::string data_dir;
    if (argc != 2)
    {
        std::cout << "Using data_dir = cwd" << std::endl;
        data_dir = std::filesystem::current_path().string();
    }
    else
    {
        data_dir = argv[1];
    }
    return data_dir;
}


int main(int argc, char** argv)
{
    auto root_data_dir = get_data_dir(argc, argv);
    for(auto&& community_type: {"/Detected_Communities/", "/True_Communities/"})
    {
        auto relative_data_dir = community_type;
        auto p_dirs = get_basename_p_dirs(root_data_dir + relative_data_dir);
        std::for_each(p_dirs.begin(), p_dirs.end(), [&](auto p_dir)
        {
            std::cout << "p_dir: " << root_data_dir + relative_data_dir + p_dir << "\n";
            compute_for_p(root_data_dir, relative_data_dir + p_dir);
        });
    }

    return 0;
}
