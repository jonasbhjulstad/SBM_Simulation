#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <Sycl_Graph/SIR_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>

std::vector<std::vector<int>> get_delta_Is(const std::vector<std::vector<State_t>> &community_state);

struct Sim_Inf_Pack
{
    Sim_Inf_Pack() = default;
    Timeseries_t<State_t> community_state;
    Timeseries_t<uint32_t> from_events;
    Timeseries_t<uint32_t> to_events;
};

struct Graph_Inf_Pack
{
    Graph_Inf_Pack() = default;
    Simseries_t<State_t> community_state;
    Simseries_t<uint32_t> from_events;
    Simseries_t<uint32_t> to_events;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::vector<uint32_t> ccm_weights;
};

Graphseries_t<uint32_t> sample_infections(
    const Graphseries_t<State_t> &community_state,
    const Graphseries_t<uint32_t> &from_events,
    const Graphseries_t<uint32_t> &to_events,
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &ccm,
    const std::vector<std::vector<uint32_t>> &ccm_weights,
    uint32_t seed, uint32_t max_infection_samples);


#endif
