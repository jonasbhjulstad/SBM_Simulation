#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <Sycl_Graph/SIR_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>

std::vector<std::vector<int>> get_delta_Is(const std::vector<std::vector<State_t>> &community_state);

struct Sim_Inf_Pack
{
    Timeseries_t<State_t> community_state;
    Timeseries_t<uint32_t> from_events;
    Timeseries_t<uint32_t> to_events;
};

struct Graph_Inf_Pack
{
    Simseries_t<State_t> community_state;
    Simseries_t<uint32_t> from_events;
    Simseries_t<uint32_t> to_events;
};




#endif
