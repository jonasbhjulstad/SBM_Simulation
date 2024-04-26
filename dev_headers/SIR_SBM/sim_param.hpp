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
  size_t Nt;
  size_t Nt_alloc;
  size_t N_I_terminate;
  size_t N_sims;
  uint32_t seed;
};


} // namespace SIR_SBM