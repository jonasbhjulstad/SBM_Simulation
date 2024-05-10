#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/sim_buffers.hpp>
#end
#src
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sycl_validate.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <SIR_SBM/population_count.hpp>
#include <SIR_SBM/exceptions.hpp>
#end
namespace SIR_SBM {


sycl::event simulation_step(sycl::queue &q, std::shared_ptr<Sim_Buffers> &SB,
                            float p_I, float p_R, uint32_t t, uint32_t t_offset,
                            sycl::event dep_event = {}) {
  auto cpy_event = state_copy(q, SB->state, t, t + 1, dep_event);
  auto rec_event = recover(q, SB->state, SB->rngs, p_R, t + 1, cpy_event);
  auto inf_event = infect(q, SB->state, SB->edges, SB->ecc, SB->contact_events,
                          SB->rngs, p_I, t + 1, t_offset, rec_event);
  return inf_event;
}

sycl::event simulation_alloc_step(sycl::queue &q,
                                  std::shared_ptr<Sim_Buffers> &SB, float p_I,
                                  float p_R, uint32_t t_offset, sycl::event dep_event = {}) {
  sycl::event step_evt = dep_event;
  auto N_steps = std::min({SB->Nt_alloc-1, SB->Nt - t_offset});
  for (int t = 0; t < N_steps; t++) {
    step_evt = simulation_step(q, SB, p_I, p_R, t, t_offset, step_evt);
  }
  auto count_evt = partition_population_count(
      q, SB->state, SB->population_count, SB->vpc, t_offset, step_evt);
  return count_evt;
}

sycl::event run_simulation(sycl::queue &q, std::shared_ptr<Sim_Buffers> &SB,
                           const Sim_Param& p) {
                            
  auto step_evt = initialize(q, SB->state, SB->rngs, p.p_I0);
  
  for (int t = 0; t < p.Nt+1; t += SB->Nt_alloc) {
    step_evt = simulation_alloc_step(q, SB, p.p_I, p.p_R, t);
    auto cpy_evt = state_copy(q, SB->state, SB->Nt_alloc - 1, 0, step_evt);
  }
  return step_evt;
}
} // namespace SIR_SBM