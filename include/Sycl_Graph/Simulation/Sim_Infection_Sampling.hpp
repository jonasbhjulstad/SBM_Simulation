#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <execution>
#include <Sycl_Graph/Utils/Common.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Dataframe/Dataframe.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/math.hpp>

void event_inf_summary(const Dataframe_t<State_t, 4> &community_state, const Dataframe_t<uint32_t, 4> &events, const std::vector<std::vector<uint32_t>> &ccms);

void event_inf_validation(const Dataframe_t<State_t, 4> &community_state, const Dataframe_t<uint32_t, 4> &events, const std::vector<std::vector<uint32_t>> &ccms);


// std::vector<uint32_t> sample_timestep_infections(const std::vector<int> &delta_Is, const std::vector<uint32_t> &from_events, const std::vector<uint32_t> &to_events, const std::vector<uint32_t> &ccm, const std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed);
Dataframe_t<int, 2> get_delta_Is(const Dataframe_t<State_t, 2> &community_state);
std::tuple<std::vector<uint32_t>,std::vector<uint32_t>> get_related_connections(size_t c_idx, const std::vector<Edge_t> &ccm);

std::vector<uint32_t> sample_timestep(const std::vector<uint32_t> &events, const std::vector<int> &delta_I, const std::vector<Edge_t> &ccm);

Dataframe_t<uint32_t, 2> sample_infections(const Dataframe_t<State_t, 2> &community_state, const Dataframe_t<uint32_t, 2> &events, const std::vector<Edge_t> &ccm, uint32_t seed);


Dataframe_t<uint32_t, 3> sample_infections(const Dataframe_t<State_t, 3> &community_state, const Dataframe_t<uint32_t, 3> &events, const std::vector<Edge_t> &ccm, uint32_t seed);

Dataframe_t<uint32_t, 4> sample_infections(const Dataframe_t<State_t, 4> &community_state, const Dataframe_t<uint32_t, 4> &events, const Dataframe_t<Edge_t, 2> &ccm, uint32_t seed);

#endif
