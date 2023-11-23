#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <Dataframe/Dataframe.hpp>
#include <QString>
#include <SBM_Database/Simulation/SIR_Types.hpp>
#include <SBM_Graph/Graph_Types.hpp>
namespace SBM_Simulation {
void event_inf_summary(
    const Dataframe::Dataframe_t<State_t, 4> &community_state,
    const Dataframe::Dataframe_t<uint32_t, 4> &events,
    const std::vector<std::vector<uint32_t>> &ccms);

void event_inf_validation(
    const Dataframe::Dataframe_t<State_t, 4> &community_state,
    const Dataframe::Dataframe_t<uint32_t, 4> &events,
    const std::vector<std::vector<uint32_t>> &ccms);

// std::vector<uint32_t> sample_timestep_infections(const std::vector<int>
// &delta_Is, const std::vector<uint32_t> &from_events, const
// std::vector<uint32_t> &to_events, const std::vector<uint32_t> &ccm, const
// std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed);
std::vector<uint32_t>
get_related_connections(size_t c_idx, const std::vector<SBM_Graph::Weighted_Edge_t> &ccm);


Dataframe::Dataframe_t<uint32_t, 2> sample_community_infections(
    uint32_t p_out_id, uint32_t graph_id, uint32_t sim_id, uint32_t community,
    const QString &control_type, const std::vector<SBM_Graph::Weighted_Edge_t> &ccm,
    uint32_t seed);
void sample_simulation_infections(uint32_t p_out_id, uint32_t graph_id,
                                  uint32_t sim_id, const QString &control_type,
                                  const std::vector<SBM_Graph::Weighted_Edge_t> &ccm,
                                  uint32_t seed);
void sample_all_infections(const QString &control_type, uint32_t seed);

} // namespace SBM_Simulation

#endif
