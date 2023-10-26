#include <SBM_Simulation/Simulation/Sim_Types.hpp>
namespace SBM_Simulation
{
    QJsonObject create_simulation_parameters(uint32_t N_pop,
                                             uint32_t p_out_id,
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
                                             float p_R,
                                             float p_I0,
                                             float p_R0)
    {
        auto toJson = [](const std::vector<uint32_t> &myVec)
        {
        QJsonArray result;
        for(int i = 0; i < myVec.size(); i++)
        {
            result.push_back(static_cast<int>(myVec[i]));
        }
        return result; };

        QJsonObject json({{"N_pop", static_cast<int>(N_pop)},
                          {"graph_id", static_cast<int>(graph_id)},
                          {"N_communities", toJson(N_communities)},
                          {"p_in", p_in},
                          {"p_out_id", static_cast<int>(p_out_id)},
                          {"p_out", p_out},
                          {"N_sims", static_cast<int>(N_sims)},
                          {"Nt", static_cast<int>(Nt)},
                          {"Nt_alloc", static_cast<int>(Nt_alloc)},
                          {"seed", static_cast<int>(seed)},
                          {"p_I_min", p_I_min},
                          {"p_I_max", p_I_max},
                          {"p_R", p_R},
                          {"p_I0", p_I0},
                          {"p_R0", p_R0}});

        return json;
    }
}
