
#include <SBM_Graph/SBM_Graph.hpp>
#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Types/Sim_Types.hpp>
#include <Sycl_Buffer_Routines/Buffer_Routines.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>
#include <tom/tom_config.hpp>
#include <chrono>

using namespace SBM_Simulation;

int main() {
  // using namespace SBM_Database;
  // Ownership of a shared_ptr()
  auto manager = Orm::DB::create({
      {"driver", tom_config::TOM_DB_DRIVER},
      {"database",
       qEnvironmentVariable("DB_DATABASE", tom_config::SQLITE3_FILENAME)},
      {"foreign_key_constraints",
       qEnvironmentVariable("DB_FOREIGN_KEYS", "true")},
      {"check_database_exists", false},
      /* Specifies what time zone all QDateTime-s will have, the overridden
         default is the Qt::UTC, set to the Qt::LocalTime or
         QtTimeZoneType::DontConvert to use the system local time. */
      {"qt_timezone", QVariant::fromValue(Qt::UTC)},
      /* Return a QDateTime with the correct time zone instead of the QString,
         only works when the qt_timezone isn't set to the DontConvert. */
      {"return_qdatetime", true},
      {"prefix", ""},
      {"prefix_indexes", false},
  });
  // project root
  std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
  std::string output_dir = root_dir + "data/";
  std::chrono::high_resolution_clock::time_point t1, t2;
  uint32_t N_pop = 10;
  uint32_t p_out_id = 0;
  uint32_t graph_id = 0;
  uint32_t N_communities = 2;
  uint32_t N_connections = SBM_Graph::complete_graph_max_edges(2);
  uint32_t N_sims = 2;
  uint32_t Nt = 20;
  uint32_t Nt_alloc = 4;
  uint32_t seed = 123;
  float p_in = 0.5f;
  float p_out = 0.1f;
  float p_I_min = 0.01f;
  float p_I_max = 0.1f;
  float p_R = 0.1f;
  float p_I0 = 0.1f;
  float p_R0 = 0.0f;

  Sim_Param p{N_pop,         p_out_id, graph_id, N_communities,
              N_connections, N_sims,   Nt,       Nt_alloc,
              seed,          p_in,     p_out,    p_I_min,
              p_I_max,       p_R,      p_I0,     p_R0};

  SBM_Graph::generate_SBM_to_db(p.to_json());

  return 0;
}
