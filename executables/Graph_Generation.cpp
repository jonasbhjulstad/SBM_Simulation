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
#include <Static_RNG/Generation/Generation.hpp>
#include <chrono>
#include <tom/tom_config.hpp>
#include <iostream>
using namespace SBM_Simulation;


// def ER_p_I(edgelist, N_pop, R0, p_R=0.1):
    // E_degree = max([len(el) for el in edgelist])/(N_pop*2)
    // return 1-np.exp(np.log(1-p_R)*R0/E_degree)

float ER_p_I(auto edgelist, uint32_t N_pop, float R0, float p_R=0.1f) {
  auto sizes = std::vector<uint32_t>(edgelist.size());
  std::transform(edgelist.begin(), edgelist.end(), sizes.begin(),
                 [](auto el) { return el.size(); });
  auto E_degree = *std::max_element(sizes.begin(), sizes.end())/(N_pop*2);
  return 1.0f-std::exp(std::log(1.0f-p_R)*R0/E_degree);
}


int main() {
  auto manager = tom_config::default_db_connection_postgres();
  uint32_t N_pop = 10;
  uint32_t N_communities = 10;
  uint32_t N_connections = SBM_Graph::complete_graph_max_edges(2);
  uint32_t N_sims = 1024;
  uint32_t Nt = 20;
  uint32_t Nt_alloc = 6;
  uint32_t seed = 123;
  float p_in = 0.5f;
  float p_I_min = 0.01f;
  float p_I_max = 0.1f;
  float p_R = 0.1f;
  float p_I0 = 0.1f;
  float p_R0 = 0.0f;
  auto Ng = 2;
  auto N_graphs_per_bulk = 2;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto end = begin;

  // std::vector<float> p_out_vec = {1.0f};
  std::vector<float> p_out_vec =
      SBM_Simulation::make_linspace(0.0f, 1.0f, 0.5f);
  auto Np = p_out_vec.size();
  auto seeds = Static_RNG::generate_seeds(Np, seed);
  SBM_Database::Sim_Param param = {N_pop, 0, 0, N_communities, N_connections, N_sims, Nt, Nt_alloc, seed, p_in, 0.0f, p_I_min, p_I_max, p_R, p_I0, p_R0};
  SBM_Database::remove_db_graphs();
  auto graph_ids = SBM_Simulation::make_iota(Ng);
  for (uint32_t p_out_id = 0; p_out_id < p_out_vec.size(); p_out_id++) {
    auto p_out = p_out_vec[p_out_id];
    param.p_out_id = p_out_id;
    param.p_out = p_out;
    begin = std::chrono::steady_clock::now();
    auto [edge_lists, node_lists] = SBM_Graph::generate_N_SBM_graphs(
        N_pop, N_communities, p_in, p_out, seeds[p_out_id], Ng);
    end = std::chrono::steady_clock::now();
    std::cout << "Graph generation time for p_out = " << p_out << " time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                        begin)
                     .count()
              << std::endl;
    for (int graph_id = 0; graph_id < Ng; graph_id++) {
      param.graph_id = graph_id;  
      SBM_Database::sim_param_upsert(param.to_json());
    }
    std::vector<uint32_t> p_outs(Ng, p_out_id);
    begin = std::chrono::steady_clock::now();
    // Orm::DB::beginTransaction();
    SBM_Database::bulk_generate_SBM_to_db(param, Ng, N_graphs_per_bulk, seeds[p_out_id], manager);
    // Orm::DB::commit();
    // auto query = manager->qtQuery();
    //   SBM_Database::SBM_Graphs_to_db(
    //       edge_lists, node_lists, p_outs, graph_ids, query);
    end = std::chrono::steady_clock::now();
    std::cout << "Graph insertion time for p_out = " << p_out << " time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                        begin)
                     .count()
              << std::endl;
  }

  return 0;
}
