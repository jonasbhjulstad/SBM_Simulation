// simulation.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_simulation_hpp
#define LZZ_SIR_SBM_LZZ_simulation_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/sim_buffers.hpp>
#define LZZ_INLINE inline
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
namespace SIR_SBM
{
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
  sycl::event simulation_step (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, float p_I, float p_R, uint32_t t, uint32_t t_offset, sycl::event dep_event = {});
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
namespace SIR_SBM
{
#line 27 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
  sycl::event simulation_alloc_step (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, float p_I, float p_R, uint32_t t_offset, sycl::event dep_event = {});
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
namespace SIR_SBM
{
#line 40 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
  sycl::event run_simulation (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, Sim_Param const & p);
}
#undef LZZ_INLINE
#endif
