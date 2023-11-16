#include <execution>

#include <SBM_Database/Graph/Generation.hpp>
#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Graph/SBM_Graph.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Utils/Math.hpp>
#include <Sycl_Buffer_Routines/Buffer_Routines.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <chrono>
#include <tom/tom_config.hpp>
#include <iostream>
using namespace SBM_Simulation;

int main() {
  auto manager = tom_config::default_db_connection_postgres();
  uint32_t N_pop = 100;
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

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto end = begin;
  auto seeds = Buffer_Routines::generate_seeds(Np, seed);

  std::vector<float> p_out_vec = {1.0f};
  // std::vector<float> p_out_vec =
      // SBM_Simulation::make_linspace(0.0f, 1.0f, 0.1f);
  SBM_Database::Sim_Param param = {N_pop, 0, 0, N_communities, N_connections, N_sims, Nt, Nt_alloc, seed, p_in, 0.0f, p_I_min, p_I_max, p_R, p_I0, p_R0};
  SBM_Database::remove_db_graphs();
  auto graph_ids = SBM_Simulation::make_iota(Ng);
  for (uint32_t p_out_id = 0; p_out_id < p_out_vec.size(); p_out_id++) {
    auto p_out = p_out_vec[p_out_id];
    param.p_out_id = p_out_id;
    param.p_out = p_out;
    auto [edge_lists, node_lists] = SBM_Graph::generate_N_SBM_graphs(
        N_pop, N_communities, p_in, p_out, seeds[p_out_id], Ng);
    for (int graph_id = 0; graph_id < Ng; graph_id++) {
      param.graph_id = graph_id;  
      SBM_Database::sim_param_upsert(param.to_json());
    }
    std::vector<uint32_t> p_outs(Ng, p_out_id);
    begin = std::chrono::steady_clock::now();
      SBM_Database::SBM_Graphs_to_db(
          edge_lists, node_lists, p_outs, graph_ids);
    end = std::chrono::steady_clock::now();
    std::cout << "p_out: " << p_out << " time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                        begin)
                     .count()
              << std::endl;
  }

  return 0;
}
