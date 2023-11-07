#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <Dataframe/Dataframe.hpp>
#include <SBM_Database/Simulation/SIR_Types.hpp>
#include <SBM_Graph/Graph_Types.hpp>

void event_inf_summary(const Dataframe::Dataframe_t<State_t, 4> &community_state, const Dataframe::Dataframe_t<uint32_t, 4> &events, const std::vector<std::vector<uint32_t>> &ccms);

void event_inf_validation(const Dataframe::Dataframe_t<State_t, 4> &community_state, const Dataframe::Dataframe_t<uint32_t, 4> &events, const std::vector<std::vector<uint32_t>> &ccms);


// std::vector<uint32_t> sample_timestep_infections(const std::vector<int> &delta_Is, const std::vector<uint32_t> &from_events, const std::vector<uint32_t> &to_events, const std::vector<uint32_t> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed);
Dataframe::Dataframe_t<int, 2> get_delta_Is(const Dataframe::Dataframe_t<State_t, 2> &community_state);
std::tuple<std::vector<uint32_t>,std::vector<uint32_t>> get_related_connections(size_t c_idx, const std::vector<Weighted_Edge_t> &ccm);

std::vector<uint32_t> sample_timestep(const std::vector<uint32_t> &events, const std::vector<int> &delta_I, const std::vector<Edge_t> &ccm);

Dataframe::Dataframe_t<uint32_t, 2> sample_infections(const Dataframe::Dataframe_t<State_t, 2> &community_state, const Dataframe::Dataframe_t<uint32_t, 2> &events, const Dataframe::Dataframe_t<Weighted_Edge_t,1> &ccm, uint32_t seed);

Dataframe::Dataframe_t<uint32_t, 3> sample_infections(const Dataframe::Dataframe_t<State_t, 3> &community_state, const Dataframe::Dataframe_t<uint32_t, 3> &events, const Dataframe::Dataframe_t<Weighted_Edge_t,1> &ccm, uint32_t seed);


#endif
