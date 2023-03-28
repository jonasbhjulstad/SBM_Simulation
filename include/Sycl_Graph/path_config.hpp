#ifndef Sycl_Graph_Path_CONFIG_HPP
#define Sycl_Graph_Path_CONFIG_HPP

#include <string>
#include <sstream>

namespace Sycl_Graph {
// const char *SYCL_GRAPH_ROOT_DIR = "";
// const char *SYCL_GRAPH_INCLUDE_DIR = "/home/man/Documents/Sycl_Graph/include";
    const char *SYCL_GRAPH_DATA_DIR = "/home/man/Documents/Old_Sycl_Graph/data";
    const char *SYCL_GRAPH_LOG_DIR = "/home/man/Documents/Old_Sycl_Graph/log";
    std::string Sim_Datapath = Sycl_Graph::SYCL_GRAPH_DATA_DIR + std::string("/SIR_sim/");
    auto community_infs_filename = [](uint32_t idx){return std::string("community_infs_" + std::to_string(idx) + ".csv");};
    auto connection_infs_filename = [](uint32_t idx){return std::string("connection_infs_" + std::to_string(idx) + ".csv");};
    auto community_recs_filename = [](uint32_t idx){return std::string("community_recs_" + std::to_string(idx) + ".csv");};
    auto community_traj_filename = [](uint32_t idx){return std::string("community_traj_" + std::to_string(idx) + ".csv");};
    auto tot_traj_filename = [](uint32_t idx){return std::string("tot_traj_" + std::to_string(idx) + ".csv");};
    auto infection_events_filename = [](uint32_t idx){return std::string("infection_events_" + std::to_string(idx) + ".csv");};
    auto connection_targets_filename = [](uint32_t idx){return std::string("connection_targets_" + std::to_string(idx) + ".csv");};
    auto connection_sources_filename = [](uint32_t idx){return std::string("connection_sources_" + std::to_string(idx) + ".csv");};
    auto p_Is_filename = [](uint32_t idx){return std::string("p_Is_" + std::to_string(idx) + ".csv");};
}
#endif
