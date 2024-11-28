#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/ticktock.hpp>
using namespace SIR_SBM;
#include <SIR_SBM/utils/filepaths.hpp>

int main() {
  TickTock t;
  t.tick();
  auto p_SBM = SBM_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  auto p_Sim = Sim_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  auto graph = generate_planted_SBM(p_SBM);
  t.tock_print();

  auto q = parse_queue(SOURCE_TEST_DIR / "simulation.yml");

  Sim_Param p;
  p_Sim.Nt = 100;
  p_Sim.N_sims = 100;
  p_Sim.seed = 10;
  Sim_Result result(p_Sim, graph);
  auto SB = Sim_Buffers(q, graph, p_Sim, result);
  SB.wait();
  auto event = initialize(q, SB.state, SB.rngs, 0.1);
  event.wait();

  return 0;
}