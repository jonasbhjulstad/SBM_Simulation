#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/filepaths.hpp>
#include <SIR_SBM/utils/ticktock.hpp>

using namespace SIR_SBM;

int main() {
  auto p_SBM = SBM_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  auto p_Sim = Sim_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(p_SBM);
  t.tock_print();

  auto q = parse_queue(SOURCE_TEST_DIR / "simulation.yml");

  Sim_Param p;
  p_Sim.Nt = 56;
  p_Sim.N_sims = 2;
  p_Sim.seed = 10;
  p_Sim.p_I0 = 0.1;
  p_Sim.p_I = 0.001;
  p_Sim.p_R = 0.1;
  Sim_Result result(p_Sim, graph);
  {
    auto SB = Sim_Buffers(q, graph, p, result);
    SB.wait();
    SB.validate(q);
    initialize(q, SB.state, SB.rngs, 0.1).wait();

    SB.validate(q);

    run_simulation(q, SB, p).wait();
  }
  auto cwd = std::filesystem::current_path();
  auto output_dir = cwd / "run_simulation_data";
  result.write(output_dir);
  return 0;
}