#ifndef SYCL_GRAPH_JSON_SETTINGS_HPP
#define SYCL_GRAPH_JSON_SETTINGS_HPP
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Utils/path_config.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
void generate_default_json(const std::string& fname)
{
    std::filesystem::create_directories(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/parameters/");
    nlohmann::json j;
    j["N_pop"] = 100;
    j["N_communities"] = 2;
    j["p_in"] = 1.0f;
    j["p_out"] = 0.0f;
    j["p_R0"] = 0.0f;
    j["p_I0"] = 0.1f;
    j["p_R"] = 1e-1f;
    j["Nt"] = 56;
    j["Nt_alloc"] = 6;
    j["p_I_max"] = 1e-3f;
    j["p_I_min"] = 1e-5f;
    j["seed"] = 283;
    j["N_graphs"] = 2;
    j["N_sims"] = 2;
    j["output_dir"] = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/p_out_0.00/";
    j["tau"] = 0.9f;
    std::ofstream o(fname);
    o << j.dump();
    o.close();
}


Sim_Param parse_json(const std::string& fname)
{
    std::ifstream i(fname);
    nlohmann::json data = nlohmann::json::parse(i);
    Sim_Param p;
    p.N_pop = data["N_pop"].get<uint32_t>();
    p.N_communities = data["N_communities"].get<uint32_t>();
    p.p_in = data["p_in"].get<float>();
    p.p_out = data["p_out"].get<float>();
    p.p_R0 = data["p_R0"].get<float>();
    p.p_I0 = data["p_I0"].get<float>();
    p.p_R = data["p_R"].get<float>();
    p.Nt = data["Nt"].get<uint32_t>();
    p.Nt_alloc = data["Nt_alloc"].get<uint32_t>();
    p.p_I_max = data["p_I_max"].get<float>();
    p.p_I_min = data["p_I_min"].get<float>();
    p.seed = data["seed"].get<uint32_t>();
    p.N_graphs = data["N_graphs"].get<uint32_t>();
    p.N_sims = data["N_sims"].get<uint32_t>();
    p.output_dir = data["output_dir"].get<std::string>();
    p.tau = data["tau"].get<float>();
    i.close();
    return p;
}

Sim_Param get_settings()
{
    const std::string json_fname = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/parameters/settings.json";
    std::ifstream i(json_fname);
    if(!i.good())
    {
        generate_default_json(json_fname);
    }
    i.close();
    return parse_json(json_fname);
}



#endif
