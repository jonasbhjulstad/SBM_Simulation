#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/ticktock.hpp>

using namespace SIR_SBM;

int main() {
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(N_pop, N_communities, p_in, p_out, seed);
  t.tock_print();

  sycl::queue q{default_queue()}; // Create a queue on the default device
  Sim_Param p;
  p.Nt = 100;
  p.N_sims = 2;
  p.seed = 10;

  Sim_Result result(p, graph);
  {
    auto SB = Sim_Buffers(q, graph, p, result);
    SB.wait();
    q.wait();
    SB.validate(q);

    initialize(q, SB.state, SB.rngs, 0.1).wait();

    SB.validate(q);

    simulation_step(q, SB, .01, 0.1, 0).wait();
    partition_population_count(q, SB.state, SB.population_count, SB.vpc).wait();
  }
  auto cwd = std::filesystem::current_path();
  auto output_dir = cwd / "simulation_step_data";
  result.write(output_dir);

  return 0;
}