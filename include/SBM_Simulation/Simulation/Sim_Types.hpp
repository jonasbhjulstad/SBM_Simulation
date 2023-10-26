#ifndef SIM_TYPES_HPP
#define SIM_TYPES_HPP
// #include <orm/db.hpp>
// #include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QList>
#include <QVariant>
#include <QVector>
#include <SBM_Simulation/Epidemiological/SIR_Types.hpp>
namespace SBM_Simulation
{
    QJsonObject create_simulation_parameters(uint32_t N_pop,
                                             uint32_t p_out_idx,
                                             uint32_t graph_id,
                                             const std::vector<uint32_t> &N_communities,
                                             float p_in,
                                             float p_out,
                                             uint32_t N_sims,
                                             uint32_t Nt,
                                             uint32_t Nt_alloc,
                                             uint32_t seed,
                                             float p_I_min,
                                             float p_I_max,
                                             float p_R = 0.1f,
                                             float p_I0 = 0.1f,
                                             float p_R0 = 0.0f);

}
#endif
