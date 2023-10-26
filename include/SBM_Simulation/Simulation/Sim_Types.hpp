#ifndef SIM_TYPES_HPP
#define SIM_TYPES_HPP
#include <orm/db.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Types.hpp>
namespace SBM_Simulation
{
    QJsonDocument create_simulation_parameters(uint32_t N_pop,
                                               uint32_t N_graphs,
                                               const std::vector<uint32_t> &N_communities,
                                               uint32_t p_out_idx,
                                               float p_in,
                                               float p_out,
                                               uint32_t N_sims,
                                               uint32_t Nt,
                                               uint32_t Nt_alloc,
                                               uint32_t seed,
                                               float p_I_min,
                                               float p_I_max,
                                               float p_R = 0.1f;
                                               float p_I0 = 0.1f;
                                               float p_R0 = 0.0f;)
    {
        QJsonObject json;
        json["N_pop"] = N_pop;
        json["N_graphs"] = N_graphs;
        json["N_communities"] = QJsonArray::fromVariantList(QVariantList::fromVector(QVector<uint32_t>::fromStdVector(N_communities)));
        json["p_in"] = p_in;
        json["p_out_idx"] = p_out_idx;
        json["p_out"] = p_out;
        json["N_sims"] = N_sims;
        json["Nt"] = Nt;
        json["Nt_alloc"] = Nt_alloc;
        json["seed"] = seed;
        json["p_I_min"] = p_I_min;
        json["p_I_max"] = p_I_max;
        json["p_R"] = p_R;
        json["p_I0"] = p_I0;
        json["p_R0"] = p_R0;
        return QJsonDocument(json);
    }

}
#endif
