#ifndef SIR_TYPES_HPP
#define SIR_TYPES_HPP
#include <array>
#include <vector>
#include <cstdint>
#include <string>
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
    uint32_t N_communities = 4;
    uint32_t N_pop = 100;
    uint32_t N_sims = 2;
    float p_in = 1.0f;
    float p_out = .0f;
    uint32_t Nt = 30;
    float p_R0 = .0f;
    float p_I0;
    float p_R;
    uint32_t N_threads = 1024;
    uint32_t seed = 238;
    uint32_t max_infection_samples = 1000;
    std::string output_dir;
};

#endif
