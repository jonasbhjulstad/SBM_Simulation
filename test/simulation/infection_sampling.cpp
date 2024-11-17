#include <SIR_SBM/epidemiological/infection_sampling.hpp>
#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/csv.hpp>
#include <SIR_SBM/utils/parameters.hpp>
#include <SIR_SBM/utils/ticktock.hpp>
#include <filesystem>
using namespace SIR_SBM;

int main() {
  TickTock t;
  t.tick();
  auto p = parse_simulation_parameters("simulation.yaml");
  auto graph = generate_planted_SBM(p);
  uint32_t N_connections = graph.N_connections();
  t.tock_print();

  Sim_Result result(p, graph);
  {
    auto SB = Sim_Buffers(q, graph, p, result);
    SB.wait();
    q.wait();
    SB.validate(q);
    initialize(q, SB.state, SB.rngs, 0.1).wait();

    SB.validate(q);

    run_simulation(q, SB, p).wait();
  }
  // result.validate();
  auto cwd = std::filesystem::current_path();
  auto output_dir = cwd / "infection_sampling_data";
  result.write(output_dir);
  // result.validate();
  Infection_Sampler sampler(p.N_sims, N_communities, N_connections, p.Nt);
  auto infections = sampler.sample_infections(result.contact_events,
                                              result.population_count, p.seed);
  write_contact_events(infections, output_dir / "infections", p.N_sims,
                       N_connections, p.Nt);
  return 0;
}