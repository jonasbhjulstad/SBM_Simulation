// simulation.hpp
//

#ifndef LZZ_simulation_hpp
#define LZZ_simulation_hpp
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/sim_buffers.hpp>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  sycl::event simulation_step (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, float p_I, float p_R, uint32_t t, uint32_t t_offset, sycl::event dep_event = {});
}
namespace SIR_SBM
{
  sycl::event simulation_alloc_step (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, float p_I, float p_R, uint32_t t_offset, sycl::event dep_event = {});
}
namespace SIR_SBM
{
  sycl::event run_simulation (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, Sim_Param const & p);
}
#undef LZZ_INLINE
#endif
