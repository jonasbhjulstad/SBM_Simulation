// sim_param.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_sim_param_hpp
#define LZZ_SIR_SBM_LZZ_sim_param_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
#include <cstdint>
#define LZZ_INLINE inline
#line 5 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
namespace SIR_SBM
{
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
  struct Sim_Param
  {
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    float p_I0;
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    float p_I;
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    float p_R;
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    size_t Nt;
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    size_t Nt_alloc;
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    size_t N_I_terminate;
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    size_t N_sims;
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_param.hpp"
    uint32_t seed;
  };
}
#undef LZZ_INLINE
#endif
