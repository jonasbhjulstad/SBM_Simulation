#ifndef SYCL_GRAPH_JSON_SETTINGS_HPP
#define SYCL_GRAPH_JSON_SETTINGS_HPP
#include <nlohmann/json.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
void generate_default_json(const std::string& fname);

Sim_Param parse_json(const std::string& fname);

#endif
