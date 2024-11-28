#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/filepaths.hpp>
#include <SIR_SBM/utils/ticktock.hpp>

using namespace SIR_SBM;

int main() {
  auto p_SBM = SBM_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  auto p_Sim = Sim_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  auto q = parse_queue(SOURCE_TEST_DIR / "simulation.yml");
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(p_SBM);
  t.tock_print();
  Sim_Result result(p_Sim, graph);
  auto SB = Sim_Buffers(q, graph, p_Sim, result);

  SB.wait();

  initialize(q, SB.state, SB.rngs, 0.1).wait();

  std::vector<Population_Count> count(
      p_SBM.N_communities * p_Sim.N_sims * p_Sim.Nt, {0, 0, 0});
  {
    auto count_buf = sycl::buffer<Population_Count, 3>{
        count.data(),
        sycl::range<3>(p_SBM.N_communities, p_Sim.N_sims, p_Sim.Nt)};
    partition_population_count(q, SB.state, count_buf, SB.vpc).wait();
  }

  auto rec_evt = recover(q, SB.state, SB.rngs, 0.1, 0);

  rec_evt.wait();

  count = std::vector<Population_Count>(
      p_SBM.N_communities * p_Sim.N_sims * p_Sim.Nt, {0, 0, 0});

  {
    auto count_buf = sycl::buffer<Population_Count, 3>{
        count.data(),
        sycl::range<3>(p_SBM.N_communities, p_Sim.N_sims, p_Sim.Nt)};
    partition_population_count(q, SB.state, count_buf, SB.vpc).wait();
  }

  return 0;
}