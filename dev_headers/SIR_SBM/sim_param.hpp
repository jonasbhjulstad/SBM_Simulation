#pragma once
#hdr
#include <cstdint>
#end
namespace SIR_SBM
{
struct Sim_Param {
  float p_I0;
  float p_I;
  float p_R;
  uint32_t Nt;
  uint32_t Nt_alloc;
  uint32_t N_I_terminate;
  uint32_t N_sims;
  uint32_t seed;
};


} // namespace SIR_SBM