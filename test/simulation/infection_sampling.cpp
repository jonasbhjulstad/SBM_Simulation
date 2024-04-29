#include <SIR_SBM/population_count.hpp>
#include <SIR_SBM/queue_select.hpp>
#include <SIR_SBM/simulation.hpp>
#include <SIR_SBM/ticktock.hpp>
#include <SIR_SBM/infection_sampling.hpp>
#include <filesystem>
using namespace SIR_SBM;

int main() {
  int N_pop = 100;
  int N_communities = 2;
  int seed = 10;
  float p_in = 1.0;
  float p_out = 1.0;
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(N_pop, N_communities,
                                                           p_in, p_out, seed);
  t.tock_print();

  sycl::queue q{sycl::cpu_selector_v}; // Create a queue on the default device
  Sim_Param p;
  p.Nt = 56;
  p.Nt_alloc = 57;
  p.N_I_terminate = 1;
  p.N_sims = 2;
  p.seed = 10;
  p.p_I0 = 0.1;
  p.p_I = 0.001;
  p.p_R = 0.1;
  Sim_Result result(p, graph);
  {
    auto SB = Sim_Buffers::make(q, graph, p, result);
    SB->wait();
    q.wait();
    SB->validate(q);
    initialize(q, SB->state, SB->rngs, 0.1).wait();

    SB->validate(q);

    run_simulation(q, SB, p).wait();
  }
  auto cwd = std::filesystem::current_path();
  auto output_dir = cwd / "infection_sampling_data";
  result.write(output_dir);
  result.validate();
  
//   auto sampled_infections = sample_infections(result.infected_count, result.population_count, seed);

  return 0;
}