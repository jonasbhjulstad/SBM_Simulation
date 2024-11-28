#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/filepaths.hpp>
#include <SIR_SBM/utils/ticktock.hpp>

using namespace SIR_SBM;

// std::pair<Sim_Buffers, std::vector<sycl::event>>
// construct_buffers(sycl::queue& q, const SBM_Graph& G, const Sim_Param& p)

int main() {
  auto p_SBM = SBM_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  auto p_Sim = Sim_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(p_SBM);
  t.tock_print();

  auto q = parse_queue(SOURCE_TEST_DIR / "simulation.yml");

  Sim_Param p;
  p_Sim.Nt = 100;
  p_Sim.N_sims = 100;
  p_Sim.seed = 10;
  Sim_Result result(p_Sim, graph);
  auto SB = Sim_Buffers(q, graph, p, result);
  SB.wait();
  // sycl::event initialize(sycl::queue& q, sycl::buffer<SIR_State, 3>& state,
  // sycl::buffer<oneapi::dpl::ranlux48>& rngs, float p_I0)
  auto event = initialize(q, SB.state, SB.rngs, 0.1);
  event.wait();

  std::vector<Population_Count> count(
      p_SBM.N_communities * p_Sim.N_sims * p_Sim.Nt, {0, 0, 0});
  {
    auto count_buf = sycl::buffer<Population_Count, 3>{
        count.data(),
        sycl::range<3>(p_SBM.N_communities, p_Sim.N_sims, p_Sim.Nt)};
    partition_population_count(q, SB.state, count_buf, SB.vpc).wait();
  }

  return 0;
}