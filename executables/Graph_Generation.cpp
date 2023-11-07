
#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Graph/SBM_Graph.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Utils/Math.hpp>
#include <Sycl_Buffer_Routines/Buffer_Routines.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>
#include <chrono>
#include <tom/tom_config.hpp>

using namespace SBM_Simulation;

int main() {
  auto manager = tom_config::default_db_connection();
  uint32_t N_pop = 10;
  uint32_t graph_id = 0;
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
  auto Np = 10;
  auto Ng = 10;

  std::vector<float> p_out_vec = SBM_Simulation::make_linspace(0.0f, 1.0f, Np);
  for (int p_out_id = 0; p_out_id < Np; p_out_id++) {
    for (int graph_id = 0; graph_id < Ng; graph_id++) {
      SBM_Database::Sim_Param p{
          N_pop,   p_out_id, graph_id, N_communities, N_connections, N_sims,
          Nt,      Nt_alloc,    seed,     p_in,          p_out_vec[p_out_id],  p_I_min,
          p_I_max, p_R,         p_I0,     p_R0};
      SBM_Database::generate_SBM_to_db(p.to_json());
    }
  }

  return 0;
}
