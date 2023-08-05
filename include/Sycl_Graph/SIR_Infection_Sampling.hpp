#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <Sycl_Graph/SIR_Types.hpp>

std::vector<uint32_t> sample_connection_infections(Inf_Sample_Data_t &z);

std::vector<uint32_t> sample_timestep_infections(const std::vector<int> &delta_Is, const std::vector<uint32_t> &from_events, const std::vector<uint32_t> &to_events, const std::vector<uint32_t> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed);
std::vector<std::vector<int>> get_delta_Is(const std::vector<std::vector<State_t>> &community_state);

std::vector<std::vector<uint32_t>> sample_infections(const std::vector<std::vector<State_t>> &community_state, const std::vector<std::vector<uint32_t>> &from_events, const std::vector<std::vector<uint32_t>> &to_events, const std::vector<uint32_t> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t seed);

#endif
