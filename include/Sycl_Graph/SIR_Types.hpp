#ifndef SIR_TYPES_HPP
#define SIR_TYPES_HPP
#include <array>
#include <vector>
#include <cstdint>
typedef std::array<uint32_t, 3> State_t;

struct Inf_Sample_Data_t
{
    uint32_t community_idx;
    uint32_t N_infected;
    uint32_t seed;
    std::vector<uint32_t> events;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> weights;
};
enum SIR_State : unsigned char {
  SIR_INDIVIDUAL_S = 0,
  SIR_INDIVIDUAL_I = 1,
  SIR_INDIVIDUAL_R = 2
};

struct Sim_Param
{
    uint32_t N_clusters = 4;
    uint32_t N_pop = 100;
    float p_in = 1.0f;
    float p_out = .0f;
    uint32_t Nt = 30;
    float p_R0 = .0f;
    float p_I0;
    float p_R;
    uint32_t sim_idx = 0;
    uint32_t max_infection_samples = 1000;
};

#endif
