#ifndef SIM_DATA_HPP
#define SIM_DATA_HPP
#include <Sycl_Graph/SIR_Types.hpp>

struct Sim_Data
{
    Sim_Data(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections);
    std::vector<std::vector<std::vector<uint32_t>>> events_to_timeseries;
    std::vector<std::vector<std::vector<uint32_t>>> events_from_timeseries;
    std::vector<std::vector<std::vector<State_t>>> state_timeseries;
    std::vector<std::vector<std::vector<uint32_t>>> connection_infections;
};

#endif
