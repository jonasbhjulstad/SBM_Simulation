
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
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

std::vector<std::vector<float>> read_p_Is(const std::string& fileToOpen)
{
    // using namespace Eigen;
    std::vector<float> matrixEntries;

    std::ifstream matrixDataFile(fileToOpen);

    std::string matrixRowString;

    std::string matrixEntry;

    int matrixRowNumber = 0;

    while (std::getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::stringstream matrixRowStringStream(matrixRowString); // convert matrixRowString that is a string to a stream variable.

        while (std::getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry)); // here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; // update the column numbers
    }
    //convert to matrix
    auto matrixColNumber = matrixEntries.size() / matrixRowNumber;
    std::vector<std::vector<float>> result(matrixRowNumber, std::vector<float>((int)matrixColNumber));
    for (int i = 0; i < matrixRowNumber; i++)
    {
        for (int j = 0; j < matrixColNumber; j++)
        {
            result[i][j] = matrixEntries[i * matrixColNumber + j];
        }
    }
    return result;
}

template <typename T>
auto duplicate(const std::vector<T>& data, auto N)
{
    std::vector<T> result(N*data.size());
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*N + j] = data[i];
        }
    }
    return result;
}

void compute_for_p(const std::string& p_dir, const std::string& simulation_subdir)
{
    auto p = Sim_Param(p_dir + "/Sim_Param.json");
    p.output_dir = p_dir + "/Validation/";
    p.simulation_subdir = simulation_subdir;
    sycl::queue q(sycl::cpu_selector_v);

    auto result_dir = p_dir + "/Excitation/";
    auto result_graph_dirs = get_graph_dirs(result_dir);

    std::vector<std::vector<std::vector<float>>> p_Is_graph(p.N_graphs);
    std::transform(result_graph_dirs.begin(), result_graph_dirs.end(), p_Is_graph.begin(), [simulation_subdir, i = 0](const auto& dir) mutable
    {
        auto res = read_p_Is(dir + simulation_subdir + "/p_Is/p_I_" + std::to_string(i) + ".csv");
        i++;
        return res;
    });
    auto p_Is = duplicate(p_Is_graph, p.N_sims);

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



    p_I_run(q, p, edge_lists, vcms, p_Is);

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
    auto p_dirs = get_p_dirs(get_data_dir(argc, argv));


    std::for_each(p_dirs.begin(), p_dirs.end(), [](auto p_dir)
    {
        compute_for_p(p_dir, "/Detected_Communities/");
        compute_for_p(p_dir, "/True_Communities/");
    });

    return 0;
}
