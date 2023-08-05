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
    std::vector<uint32_t> connection_events;
    std::vector<uint32_t> connection_indices;
    std::vector<uint32_t> connection_weights;
};
enum SIR_State : unsigned char {
  SIR_INDIVIDUAL_S = 0,
  SIR_INDIVIDUAL_I = 1,
  SIR_INDIVIDUAL_R = 2
};

#endif
