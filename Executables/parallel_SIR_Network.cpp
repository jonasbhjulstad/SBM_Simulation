
//enable tbb debug
#define TBB_USE_DEBUG 1

#include <Sycl_Graph/Utils/path_config.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Utils/json_settings.hpp>
#include <execution>
#include <iomanip>
#include <chrono>
template <typename T>
std::vector<T> merge_vectors(const std::vector<std::vector<T>> &vectors)
{
    std::vector<T> merged;
    uint32_t size = 0;
    for(int i = 0; i < vectors.size(); i++)
    {
        size += vectors[i].size();
    }
    merged.reserve(size);
    for (auto &v : vectors)
    {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}

auto generate_graphs(const std::vector<Sim_Param>& ps, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::vector<uint32_t> seeds(ps.size());
    std::generate(seeds.begin(), seeds.end(), [&gen]()
    {
        return gen();
    });
    std::vector<std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>, std::vector<std::vector<uint32_t>>>> graphs(ps.size());
    std::transform(std::execution::par_unseq, ps.begin(), ps.end(), seeds.begin(),  graphs.begin(), [&](const auto& p, auto s)
    {

        return generate_N_SBM_graphs(p.N_pop, p.N_communities, p.p_in, p.p_out, p.seed, p.N_graphs);
    });
    return graphs;
}

Sim_Param get_settings()
{
    const std::string json_fname = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/parameters/settings.json";
    //check if exists
    std::ifstream i(json_fname);
    if(!i.good())
    {
        generate_default_json(json_fname);
    }
    i.close();
    return parse_json(json_fname);
}

std::vector<Sim_Param> create_sim_param(const std::vector<sycl::queue>& qs)
{
    auto p = get_settings();
    std::vector<Sim_Param> ps(qs.size(), p);
    std::mt19937 gen(p.seed);
    std::vector<uint32_t> seeds(ps.size());
    std::generate(seeds.begin(), seeds.end(), [&gen]()
    {
        return gen();
    });
    for(int i = 0; i < qs.size(); i++)
    {
        auto device = qs[i].get_device();
        ps[i].wg_range = p.N_sims;
        ps[i].compute_range = p.N_graphs*p.N_sims;
        ps[i].seed = seeds[i];
        ps[i].local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        ps[i].global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    }
    return ps;
}



int main(int argc, char **argv)
{
    //parse argv which is N_sims, N_communities, N_pop

    std::chrono::high_resolution_clock::time_point t1, t2;

    auto N_wg = std::stoi(argv[1]);
    uint32_t N_communities = std::stoi(argv[2]);
    uint32_t N_pop = std::stoi(argv[3]);
    uint32_t N_graphs = std::stoi(argv[4]);
    //get all gpus
    std::vector<sycl::queue> qs;
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto &d : devices)
    {
        qs.emplace_back(d);
    }
    uint32_t seed = 283;
    auto ps = create_sim_param(qs);

    t1 = std::chrono::high_resolution_clock::now();

    auto graphs = generate_graphs(ps, seed);

    std::vector<Sim_Buffers> buffers;
    std::vector<std::tuple<std::decay_t<decltype(graphs[0])>, sycl::queue, Sim_Param>> buffer_params;
    for(int i = 0; i < graphs.size(); i++)
    {
        buffer_params.push_back(std::make_tuple(graphs[i], qs[i], ps[i]));
    }

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_lists_flat(std::get<0>(graphs[0]).size());
    std::vector<std::vector<uint32_t>> vcms(std::get<0>(graphs[0]).size());
    std::vector<std::vector<uint32_t>> ecms(std::get<0>(graphs[0]).size());
    for(int i = 0; i < std::get<0>(graphs[0]).size(); i++)
    {    //flatten edge_lists
        auto& e_list = std::get<0>(graphs[0])[i];
        auto& nodelists = std::get<1>(graphs[0])[i];

        edge_lists_flat[i] = merge_vectors(e_list);
        vcms[i] = create_vcm(nodelists);
        ecms[i] = std::get<2>(graphs[0])[i];
    }

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Generate graphs: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    t1 = t2;



    auto b = Sim_Buffers::make(qs[0], ps[0], edge_lists_flat, vcms, ecms, {});
    std::for_each(qs.begin(), qs.end(), [](auto& q){q.wait();});
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Make buffers: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
    run(qs[0], ps[0], b);


    return 0;
}
