#include <SBM_Database/Graph/Generation.hpp>
#include <SBM_Graph/Complete_Graph.hpp>
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <SBM_Simulation/Utils/Math.hpp>
#include <Static_RNG/Generation/Generation.hpp>
#include <doctest/doctest.h>

bool check_3d_range(uint32_t N0, uint32_t N1, uint32_t N2, const auto& buf)
{
    auto r = buf.get_range();
    return (N0 == r[0] && N1 == r[1] && N2 == r[2]);
}

void check_buffer_ranges(const SBM_Simulation::Sim_Buffers& b, const SBM_Database::Sim_Param& p)
{
    CHECK(check_3d_range(p.N_sims, p.Nt_alloc+1, p.N_pop*p.N_communities, b.vertex_state));
    CHECK(check_3d_range(p.N_sims, p.Nt_alloc, p.N_connections, b.accumulated_events));
    CHECK(check_3d_range(p.N_sims, p.Nt, p.N_connections, b.p_Is));
    CHECK(b.rngs.get_range()[0] == p.N_sims);
    CHECK(b.ecm.get_range()[0] == b.edges.get_range()[0]);
    CHECK(b.vpm.get_range()[0] == p.N_pop*p.N_communities);
    CHECK(check_3d_range(p.N_sims, p.Nt_alloc+1, p.N_communities, b.community_state));
}

TEST_CASE("Sim_Buffers") {
  uint32_t N_pop = 10;
  uint32_t N_communities = 2;
  uint32_t N_connections = SBM_Graph::complete_graph_max_edges(2);
  uint32_t N_sims = 2;
  uint32_t Nt = 20;
  uint32_t Nt_alloc = 4;
  uint32_t seed = 123;
  float p_in = 0.5f;
  float p_I_min = 0.01f;
  float p_I_max = 0.1f;
  float p_R = 0.1f;
  float p_I0 = 0.1f;
  float p_R0 = 0.0f;
  auto Np = 1;
  auto Ng = 1;
  float p_out = 0.5f;
  auto seeds = Static_RNG::generate_seeds(Np, seed);

  std::vector<float> p_out_vec =
      SBM_Simulation::make_linspace(0.0f, 1.0f, 0.1f);
  SBM_Database::Sim_Param param = {
      N_pop, 0,    0,    N_communities, N_connections, N_sims, Nt,   Nt_alloc,
      seed,  p_in, 0.0f, p_I_min,       p_I_max,       p_R,    p_I0, p_R0};
  auto p_out_id = 0;
  auto [edge_lists, node_lists] = SBM_Graph::generate_N_SBM_graphs(
      N_pop, N_communities, p_in, p_out, seeds[p_out_id], 1);
  auto graph_id = 0;
  SBM_Database::drop_graph_tables();
  SBM_Database::SBM_Graph_to_db(edge_lists[graph_id], node_lists[graph_id],
                                   p_out_id, graph_id);

  auto q = sycl::queue(sycl::default_selector_v);
  auto b = SBM_Simulation::Sim_Buffers(q, param, "Community");
  check_buffer_ranges(b, param);
}