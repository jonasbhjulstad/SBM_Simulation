#ifndef SIM_TYPES_HPP
#define SIM_TYPES_HPP
#include <QJsonObject>
#include <SBM_Simulation/Types/SIR_Types.hpp>
namespace SBM_Simulation
{

    struct Sim_Param
    {
        uint32_t N_pop;
        uint32_t p_out_id;
        uint32_t graph_id;
        uint32_t N_communities;
        uint32_t N_connections;
        uint32_t N_sims;
        uint32_t Nt;
        uint32_t Nt_alloc;
        uint32_t seed;
        float p_in;
        float p_out;
        float p_I_min;
        float p_I_max;
        float p_R = 0.1f;
        float p_I0 = 0.1f;
        float p_R0 = 0.0f;
        QJsonObject to_json() const;
        static Sim_Param from_json(QJsonObject json);
    };

}
#endif
