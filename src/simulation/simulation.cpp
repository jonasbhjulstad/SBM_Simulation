
#include <SIR_SBM/epidemiological/epidemiological.hpp>
#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/sycl/sycl_routines.hpp>
#include <SIR_SBM/sycl/sycl_validate.hpp>
#include <SIR_SBM/utils/exceptions.hpp>

namespace SIR_SBM {

sycl::event simulation_step(sycl::queue &q, Sim_Buffers &SB,
                            float p_I, float p_R, uint32_t t,
                            sycl::event dep_event ) {
  auto cpy_event = state_copy(q, SB.state, t, t + 1, dep_event);
  auto rec_event = recover(q, SB.state, SB.rngs, p_R, t + 1, cpy_event);
  auto inf_event = infect(q, SB.state, SB.edges, SB.ecc, SB.contact_events,
                          SB.rngs, p_I, t + 1, rec_event);
  return inf_event;
}



sycl::event run_simulation(sycl::queue &q, Sim_Buffers &SB,
                           const Sim_Param &p) {

  auto step_evt = initialize(q, SB.state, SB.rngs, p.p_I0);

  for (int t = 0; t < p.Nt; t++) {
    step_evt = simulation_step(q, SB, p.p_I, p.p_R, t, step_evt);
  }
  auto count_evt = partition_population_count(
    q, SB.state, SB.population_count, SB.vpc, step_evt);
  return step_evt;
}

} // namespace SIR_SBM