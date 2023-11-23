#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <Static_RNG/Generation/Generation.hpp>
#include <SBM_Graph/Complete_Graph.hpp>
#include <tom/tom_config.hpp>
int main() {
  using namespace SBM_Database;
  auto manager = tom_config::default_db_connection_postgres();
  uint32_t N_pop = 400;
  uint32_t N_communities = 2;
  uint32_t N_connections = SBM_Graph::complete_graph_max_edges(N_communities);
  uint32_t N_sims = 2;
  uint32_t Nt = 20;
  uint32_t Nt_alloc = 2;
  uint32_t seed = 123;
  float p_in = 0.5f;
  float p_I_min = 0.0001f;
  float p_I_max = 0.001f;
  float p_R = 0.1f;
  float p_I0 = 0.1f;
  float p_R0 = 0.0f;
  auto Np = 10;
  auto Ng = 10;

  auto seeds = Static_RNG::generate_seeds(Np, seed);

  Sim_Param p = {N_pop,   0,        0,    N_communities, N_connections, N_sims,
                 Nt,      Nt_alloc, seed, p_in,          0.0f,          p_I_min,
                 p_I_max, p_R,      p_I0, p_R0};

  for (int i = 0; i < Ng; i++) {
    p.graph_id = i;
    for (int j = 0; j < Np; j++) {
      p.p_out_id = j;
      sim_param_upsert(p.to_json());
    }
  }
}