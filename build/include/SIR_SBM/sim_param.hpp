// sim_param.hpp
//

#ifndef LZZ_sim_param_hpp
#define LZZ_sim_param_hpp
#include <cstdint>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  struct Sim_Param
  {
    float p_I0;
    float p_I;
    float p_R;
    size_t Nt;
    size_t Nt_alloc;
    size_t N_I_terminate;
    size_t N_sims;
    uint32_t seed;
  };
}
#undef LZZ_INLINE
#endif
