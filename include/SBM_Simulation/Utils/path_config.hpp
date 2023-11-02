#ifndef SBM_Simulation_Path_CONFIG_HPP
#define SBM_Simulation_Path_CONFIG_HPP

#include <string>
#include <sstream>
#include <cstdint>

namespace SBM_Simulation {
// const char *SBM_SIMULATION_ROOT_DIR = "";
// const char *SBM_SIMULATION_INCLUDE_DIR = "/home/man/Documents/SBM_Simulation/include";
    const char *SBM_SIMULATION_DATA_DIR = "/home/deb/Documents/SBM_Simulation/data";
    const char *SBM_SIMULATION_LOG_DIR = "/home/deb/Documents/SBM_Simulation/log";
    std::string Sim_Datapath = SBM_Simulation::SBM_SIMULATION_DATA_DIR + std::string("/SIR_sim/");
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
