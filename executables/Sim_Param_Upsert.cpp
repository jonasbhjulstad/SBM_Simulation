#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Graph/Complete_Graph.hpp>
#include <Static_RNG/Generation/Generation.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <SBM_Simulation/Utils/P_I_Generation.hpp>
#include <tom/tom_config.hpp>



float ER_p_I(auto edge_count, uint32_t N_pop, float R0, float p_R = 0.1f) {
  uint32_t E_degree = edge_count / (N_pop * 2);
  return 1.0f - std::exp(std::log(1.0f - p_R) * R0 / E_degree);
}

int main() {
  using namespace SBM_Database;
  using namespace SBM_Simulation;
  auto manager = tom_config::default_db_connection_postgres();

  float R0_min = .8f;
  float R0_max = 1.4f;

  // get value
  auto Np = get_N_p_out("vertex_partition_map");
  auto Ng = get_N_graphs("vertex_partition_map");
  Orm::DB::table("p_Is_excitation")->remove();
  for(int i = 0; i < Np; i++)
  {
    for(int j = 0; j < Ng; j++)
    {
      auto p = sim_param_read(i, j);
      auto query = Orm::DB::unprepared(
          "SELECT COUNT(*) FROM edgelists WHERE p_out = " + QString::number(i) + " AND graph = " + QString::number(j));
      query.next();
      auto edge_count = query.value(0).toInt();
      p.p_out_id = i;
      p.graph_id = j;
      p.p_I_min = ER_p_I(edge_count, p.N_pop*p.N_communities, R0_min);
      p.p_I_max = ER_p_I(edge_count, p.N_pop*p.N_communities, R0_max);
      p.N_sims = 2;
      sim_param_upsert(p);
      generate_insert_p_Is(p, "p_Is_excitation", "Community");
    }
  }
// sycl::buffer<float, 3>
// generate_p_Is_excitation(sycl::queue &q, const SBM_Database::Sim_Param &p,
//                          const QString &control_type);
// sycl::buffer<float, 3>
// generate_insert_p_Is(sycl::queue &q, const SBM_Database::Sim_Param &p,
//                      const QString &table_name, const QString &control_type,
//                      const QString &tmp_dir = "/tmp/");

}