#ifndef SIR_INFECTION_SAMPLING_HPP
#define SIR_INFECTION_SAMPLING_HPP
#include <Dataframe/Dataframe.hpp>
#include <QString>
#include <SBM_Database/Simulation/SIR_Types.hpp>
#include <SBM_Graph/Graph_Types.hpp>
namespace SBM_Simulation {

// std::vector<uint32_t> sample_timestep_infections(const std::vector<int>
// &delta_Is, const std::vector<uint32_t> &from_events, const
// std::vector<uint32_t> &to_events, const std::vector<uint32_t> &ccm, const
// std::vector<uint32_t> &ccm_weights, uint32_t N_connections, uint32_t seed);

void sample_all_infections(const QString &control_type,
                           const QString &regression_type, uint32_t seed);

} // namespace SBM_Simulation

#endif
