#ifndef Sycl_Graph_Path_CONFIG_HPP
#define Sycl_Graph_Path_CONFIG_HPP

#include <string>
#include <sstream>
#include <cstdint>

namespace Sycl_Graph {
// const char *SYCL_GRAPH_ROOT_DIR = "";
// const char *SYCL_GRAPH_INCLUDE_DIR = "/home/man/Documents/Sycl_Graph/include";
    const char *SYCL_GRAPH_DATA_DIR = "/home/man/Documents/ER_Bernoulli_Robust_MPC/data";
    const char *SYCL_GRAPH_LOG_DIR = "/home/man/Documents/ER_Bernoulli_Robust_MPC/log";
    std::string Sim_Datapath = Sycl_Graph::SYCL_GRAPH_DATA_DIR + std::string("/SIR_sim/");
    auto community_infs_filename = [](uint32_t idx){return std::string("community_infections_" + std::to_string(idx) + ".csv");};
    auto connection_infs_filename = [](uint32_t idx){return std::string("connection_infections_" + std::to_string(idx) + ".csv");};
    // auto community_recs_filename = [](uint32_t idx){return std::string("community_recs_" + std::to_string(idx) + ".csv");};
    auto community_traj_filename = [](uint32_t idx){return std::string("community_trajectory_" + std::to_string(idx) + ".csv");};
    // auto tot_traj_filename = [](uint32_t idx){return std::string("tot_traj_" + std::to_string(idx) + ".csv");};
    auto infection_events_filename = [](uint32_t idx){return std::string("connection_events_" + std::to_string(idx) + ".csv");};
    // auto connection_targets_filename = [](uint32_t idx){return std::string("connection_targets_" + std::to_string(idx) + ".csv");};
    // auto connection_sources_filename = [](uint32_t idx){return std::string("connection_sources_" + std::to_string(idx) + ".csv");};
    auto connection_community_map_filename = [](){return std::string("connection_community_map.csv");};
    auto p_Is_filename = [](uint32_t idx){return std::string("p_Is_" + std::to_string(idx) + ".csv");};
}
#endif
