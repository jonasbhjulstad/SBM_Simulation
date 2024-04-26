#include <SIR_SBM/queue_select.hpp>
#include <SIR_SBM/simulation.hpp>
#include <SIR_SBM/ticktock.hpp>
using namespace SIR_SBM;

// std::pair<Sim_Buffers, std::vector<sycl::event>>
// construct_buffers(sycl::queue& q, const SBM_Graph& G, const Sim_Param& p)

int main() {
  int N_pop = 100;
  int N_communities = 2;
  int seed = 10;
  float p_in = 0.5;
  float p_out = 1.0;
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(N_pop, N_communities,
                                                           p_in, p_out, seed);
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
  auto event = initialize(q, SB->state, SB->rngs, 0.1);
  event.wait();

  return 0;
}