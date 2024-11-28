#include <SIR_SBM/graph/graph.hpp>
#include <SIR_SBM/math/combination.hpp>
#include <SIR_SBM/sycl/queue_select.hpp>
#include <SIR_SBM/utils/exceptions.hpp>
#include <SIR_SBM/utils/ticktock.hpp>
using namespace SIR_SBM;
#include <SIR_SBM/utils/filepaths.hpp>

int main() {
  auto p_SBM = SBM_Param::parse(SOURCE_TEST_DIR / "simulation.yml");
  TickTock t;
  t.tick();
  auto graph = generate_planted_SBM(p_SBM);
  t.tock_print();
  auto N_pop = graph.N_vertices();
  uint32_t N_connections = graph.N_connections();

  auto N_in = bipartite_max_edges(N_pop, N_pop) * p_SBM.N_communities;
  auto N_out =
      bipartite_max_edges(N_pop, N_pop) * n_choose_k(p_SBM.N_communities, 2);

  t.tick();
  auto graph_out = generate_planted_SBM(p_SBM);
  throw_if(graph_out.N_edges() != N_out, "Wrong number of planted out edges");
  t.tock_print();
  t.tick();
  auto graph_in = generate_planted_SBM(p_SBM);
  int N_edges = graph_in.N_edges();
  throw_if(graph_in.N_edges() != N_in, "Wrong number of planted in edges");
  t.tock_print();
  return 0;
}