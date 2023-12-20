#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Graph/Complete_Graph.hpp>
#include <Static_RNG/Generation/Generation.hpp>
#include <SBM_Database/Simulation/Size_Queries.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <tom/tom_config.hpp>

float ER_p_I(auto edge_count, uint32_t N_pop, float R0, float p_R = 0.1f)
{
  uint32_t E_degree = edge_count / (N_pop * 2);
  return 1.0f - std::exp(std::log(1.0f - p_R) * R0 / E_degree);
}

int main()
{
  using namespace SBM_Database;
  auto manager = tom_config::default_db_connection_postgres();

  float R0_min = .8f;
  float R0_max = 1.4f;
  auto Np = get_N_p_out("simulation_parameters");
  auto Ng = get_N_graphs("vertex_partition_map");
  Orm::DB::table("p_Is_excitation")->remove();
  std::ofstream f("p_Is.csv");
  auto single_graph_write = [&f](const Sim_Param &p)
  {
    const std::vector<float> &p_Is = Buffer_Routines::generate_floats(p.N_sims * p.Nt * p.N_connections, p.p_I_min, p.p_I_max, 8, p.seed);
    auto lin_idx = 0;
    for (int sim_id = 0; sim_id < p.N_sims; sim_id++)
    {
      for (int t = 0; t < p.Nt; t++)
      {
        for (int con_id = 0; con_id < p.N_connections; con_id++)
        {
          f << p.p_out_id << "," << p.graph_id << "," << sim_id << "," << t << "," << con_id << "," << p_Is[lin_idx] << ",Community" << "\n";
          lin_idx++;
        }
      }
    }
  };

  for (int i = 0; i < Np; i++)
  {
    for (int j = 0; j < Ng; j++)
    {
      auto p = sim_param_read(i, j);
      auto query = Orm::DB::unprepared(
          "SELECT COUNT(*) FROM edgelists WHERE p_out = " + QString::number(i) + " AND graph = " + QString::number(j));
      query.next();
      auto edge_count = query.value(0).toInt();
      p.p_out_id = i;
      p.graph_id = j;
      p.p_I_min = ER_p_I(edge_count, p.N_pop * p.N_communities, R0_min);
      p.p_I_max = ER_p_I(edge_count, p.N_pop * p.N_communities, R0_max);
      p.N_sims = 2;
      sim_param_upsert(p);
      single_graph_write(p);
    }
  }
}