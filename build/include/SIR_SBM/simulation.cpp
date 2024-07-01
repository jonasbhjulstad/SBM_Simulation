// simulation.cpp
//

#include "simulation.hpp"
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sycl_validate.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <SIR_SBM/population_count.hpp>
#include <SIR_SBM/exceptions.hpp>
#define LZZ_INLINE inline
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
namespace SIR_SBM
{
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
  sycl::event simulation_step (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, float p_I, float p_R, uint32_t t, uint32_t t_offset, sycl::event dep_event)
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
                                                        {
  auto cpy_event = state_copy(q, SB->state, t, t + 1, dep_event);
  auto rec_event = recover(q, SB->state, SB->rngs, p_R, t + 1, cpy_event);
  auto inf_event = infect(q, SB->state, SB->edges, SB->ecc, SB->contact_events,
                          SB->rngs, p_I, t + 1, t_offset, rec_event);
  return inf_event;
}
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
namespace SIR_SBM
{
#line 27 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
  sycl::event simulation_alloc_step (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, float p_I, float p_R, uint32_t t_offset, sycl::event dep_event)
#line 29 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
                                                                                            {
  sycl::event step_evt = dep_event;
  auto N_steps = std::min({SB->Nt_alloc-1, SB->Nt - t_offset});
  for (int t = 0; t < N_steps; t++) {
    step_evt = simulation_step(q, SB, p_I, p_R, t, t_offset, step_evt);
  }
  auto count_evt = partition_population_count(
      q, SB->state, SB->population_count, SB->vpc, t_offset, step_evt);
  return count_evt;
}
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
namespace SIR_SBM
{
#line 40 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
  sycl::event run_simulation (sycl::queue & q, std::shared_ptr <Sim_Buffers> & SB, Sim_Param const & p)
#line 41 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//simulation.hpp"
                                               {
                            
  auto step_evt = initialize(q, SB->state, SB->rngs, p.p_I0);
  
  for (int t = 0; t < p.Nt+1; t += SB->Nt_alloc) {
    step_evt = simulation_alloc_step(q, SB, p.p_I, p.p_R, t);
    auto cpy_evt = state_copy(q, SB->state, SB->Nt_alloc - 1, 0, step_evt);
  }
  return step_evt;
}
}
#undef LZZ_INLINE
