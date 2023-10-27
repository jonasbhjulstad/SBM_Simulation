#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <orm/db.hpp>
#include <SBM_Graph/Database/Graph_Tables.hpp>
#include <SBM_Graph/Utils/Math.hpp>
void create_graph_db(auto cwd) {
  std::system(
      ("sqlite3 " + cwd
       + "/HelloWorld.sqlite3 'create table if not exists posts(id INTEGER NOT NULL PRIMARY KEY "
         "AUTOINCREMENT, name VARCHAR NOT NULL); insert into posts values(1, 'First Post'); insert "
         "into posts values(2, 'Second Post'); select * from posts;'")
          .c_str());
}

void remove_graph_db(auto cwd) { std::system(("rm -rf" + cwd + "HelloWorld.sqlite3").c_str()); }


void create_edgelist_table() {
  using Orm::DB;
  DB::statement("CREATE TABLE IF NOT EXISTS edgelists(p_out INTEGER NOT NULL,graph INTEGER NOT NULL,edge INTEGER NOT NULL,'from' INTEGER NOT NULL,'to' INTEGER NOT NULL, weight INTEGER NOT NULL, PRIMARY KEY(p_out, graph, edge))");

}

void drop_edgelist_table()
{
  using Orm::DB;
  DB::statement("DROP TABLE IF EXISTS edgelists");
}

void edgelist_test() { create_edgelist_table();
  using Orm::DB;
  drop_edgelist_table();
  create_edgelist_table();
  DB::table("edgelists")->insert({{"p_out", 0}, {"graph", 0}, {"edge", 0},{"from", 0}, {"to", 0}, {"weight", 0}});


}
TEST_CASE("graph_tables") {
  using Orm::DB;
  auto cwd = std::filesystem::current_path().generic_string();
  remove_graph_db(cwd);
  create_graph_db(cwd);
  edgelist_test();

  int a = 1;
}
TEST_CASE("graph_tables")
{
  using namespace SBM_Graph;
  using Orm::DB;
  auto cwd = std::filesystem::current_path().generic_string();
  drop_graph_tables();
  create_graph_tables();
  auto from_vec = SBM_Graph::make_iota(100);
  auto to_vec = from_vec;
  std::reverse(to_vec.begin(), to_vec.end());
  std::vector<uint32_t> weights(100, 1);
  uint32_t p_out = 1;
  uint32_t graph = 1;

  edgelist_insert(p_out, graph, from_vec, to_vec, weights);
  auto a = 1;
}

TEST_CASE("select")
{
  using namespace SBM_Graph;
  using Orm::DB;
  auto cwd = std::filesystem::current_path().generic_string();
  drop_graph_tables();
  create_graph_tables();
  auto from_vec = make_iota(100);
  auto to_vec = from_vec;
  std::reverse(to_vec.begin(), to_vec.end());
  std::vector<uint32_t> weights(100, 1);

  uint32_t p_out = 1;
  uint32_t graph = 1;
  edgelist_insert(p_out, graph, from_vec, to_vec, weights);
  auto from_q = DB::table("edgelists")->where({{"p_out", p_out}, {"graph", graph}}).select("from").get();
  auto a = 1;
}

TEST_CASE("QVariant")
{
  Orm::DB::statement("CREATE TABLE IF NOT EXISTS array_test(array INTEGER[3])");
  Orm::DB::table("array_test")->insert({{"array", QVariant::fromValue(QVector<int>{1, 2, 3})}});
  auto a = 1;
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
