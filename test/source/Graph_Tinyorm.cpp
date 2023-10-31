#include <doctest/doctest.h>

#include <SBM_Graph/Database/Generation.hpp>
#include <SBM_Graph/Database/Graph_Tables.hpp>
#include <SBM_Graph/Utils/Math.hpp>
#include <cstdlib>
#include <filesystem>
#include <orm/db.hpp>

void db_create()
{
  auto gt_str = SBM_Graph::create_graph_tables_str();

  std::system(("sqlite3 HelloWorld.sqlite3 '" + gt_str + "'").c_str());
}

void db_drop()
{
  std::system("rm HelloWorld.sqlite3");
}


// TEST_CASE("edgelists_tests")
// {
//   using namespace SBM_Graph;
//   using Orm::DB;
//   auto cwd = std::filesystem::current_path().generic_string();
//   auto from_vec = make_iota(100);
//   auto to_vec = from_vec;
//   std::reverse(to_vec.begin(), to_vec.end());
//   std::vector<uint32_t> weights(100, 1);

//   uint32_t p_out = 1;
//   uint32_t graph = 1;
//   {
//   edgelist_insert(p_out, graph, from_vec, to_vec, weights);
//   edgelist_insert(2, 4, from_vec, to_vec, weights);
//   }
//   read_edgelist(p_out, graph);
// }

TEST_CASE("vcm_test")
{
  using namespace SBM_Graph;
  using Orm::DB;
  auto cwd = std::filesystem::current_path().generic_string();
  db_drop();
  db_create();

  auto vcm = make_iota(100);
  vcm_insert(1, 1, vcm);
  vcm_insert(0, 0, vcm);

  vcm = vcm_read(1, 1);
}

TEST_CASE("ecm_test")
{
  using namespace SBM_Graph;
  using Orm::DB;
  auto cwd = std::filesystem::current_path().generic_string();
  db_drop();
  db_create();


  auto ecm = make_iota(100);
  ecm_insert(1, 1, ecm);
  ecm_insert(0, 0, ecm);

  ecm = ecm_read(1, 1);
}


TEST_CASE("graph_tables")
{

  using Orm::DB;
  using namespace SBM_Graph;
  db_drop();
  db_create();

  auto cwd = std::filesystem::current_path().generic_string();
  auto p = QJsonObject();
  p["N_communities"] = 2;
  p["N_pop"] = 10;
  p["N_graphs"] = 2;
  p["N_communities"] = 2;
  p["p_in"] = 0.5f;
  p["p_out"] = 0.1f;
  p["N_sims"] = 2;
  p["Nt"] = 56;
  p["Nt_alloc"] = 20;
  p["seed"] = 234;
  p["p_I_min"] = 0.1f;
  p["p_I_max"] = 0.2f;
  p["p_out_id"] = 0;
  p["p_R"] = 0.1f;
  p["p_I0"] = 0.1f;
  p["p_R0"] = 0.0f;

  generate_SBM_to_db(p);

  int a = 1;
}
// TEST_CASE("simulation_tables")
// {
//   using namespace SBM_Graph;
//   using Orm::DB;
//   auto cwd = std::filesystem::current_path().generic_string();
//   drop_simulation_tables();
//   create_simulation_tables();
//   auto a = 1;
// }

// TEST_CASE("edgelists")
// {
//   edgelist_test();
// }
