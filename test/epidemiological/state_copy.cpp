#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/simulation/simulation.hpp>
#include <SIR_SBM/utils/ticktock.hpp>

using namespace SIR_SBM;

int main() {
  int N_pop = 100;
  int N_communities = 2;
  int seed = 10;
  float p_in = 0.5;
  float p_out = 1.0;
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(N_pop, N_communities, p_in, p_out, seed);
  t.tock_print();

  sycl::queue q{
      SIR_SBM::default_queue()}; // Create a queue on the default device
  Sim_Param p;
  p.Nt = 100;
  p.Nt_alloc = 100;
  p.N_I_terminate = 1;
  p.N_sims = 100;
  p.seed = 10;
  Sim_Result result(p, graph);
  auto SB = Sim_Buffers::make(q, graph, p, result);
  SB->wait();

  initialize(q, SB->state, SB->rngs, 0.1).wait();

  std::vector<Population_Count> count(N_communities * p.N_sims * p.Nt,
                                      {0, 0, 0});
  {
    auto count_buf = sycl::buffer<Population_Count, 3>{
        count.data(), sycl::range<3>(N_communities, p.N_sims, p.Nt)};
    partition_population_count(q, SB->state, count_buf, SB->vpc, 0).wait();
  }

  state_copy(q, SB->state, 0, 1).wait();

  count =
      std::vector<Population_Count>(N_communities * p.N_sims * p.Nt, {0, 0, 0});

  {
    auto count_buf = sycl::buffer<Population_Count, 3>{
        count.data(), sycl::range<3>(N_communities, p.N_sims, p.Nt)};
    partition_population_count(q, SB->state, count_buf, SB->vpc, 0).wait();
  }

  return 0;
}