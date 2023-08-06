#ifndef SIR_SIMULATION_HPP
#define SIR_SIMULATION_HPP
#include <cstdint>
#include <vector>
#include <tuple>
#include <string>
struct Sim_Param
{
    uint32_t N_clusters = 4;
    uint32_t N_pop = 100;
    float p_in = 1.0f;;
    float p_out = .0f;
    uint32_t Nt = 30;
    float p_R0 = .0f;
    float p_I0;
    float p_R;
    uint32_t sim_idx = 0;
    uint32_t seed = 47;
};
void excite_simulate(const Sim_Param& p, const std::vector<uint32_t>& vcm, const std::vector<std::pair<uint32_t, uint32_t>>& edge_list, float p_I_min, float p_I_max, const std::string output_dir = "./");

#endif
