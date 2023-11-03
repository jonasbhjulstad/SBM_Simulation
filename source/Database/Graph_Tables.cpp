#include <SBM_Graph/Database/Graph_Tables.hpp>
// #include <QVariant>
namespace SBM_Graph {
void create_graph_tables() {
  Orm::DB::statement("CREATE TABLE IF NOT EXISTS edgelists(p_out INTEGER NOT "
                     "NULL,graph INTEGER NOT NULL,edge "
                     "INTEGER NOT NULL,\"from\" INTEGER NOT NULL,\"to\" INTEGER "
                     "NOT NULL, weight INTEGER NOT NULL, "
                     "PRIMARY KEY(p_out, graph, edge))");
  Orm::DB::commit();

  Orm::DB::statement(
      "CREATE TABLE IF NOT EXISTS connection_community_map(p_out INTEGER NOT "
      "NULL,graph INTEGER "
      "NOT NULL,edge INTEGER NOT NULL,\"from\" INTEGER NOT NULL,\"to\" INTEGER NOT "
      "NULL, weight "
      "INTEGER NOT NULL, PRIMARY KEY(p_out, graph, edge))");
  Orm::DB::commit();

  Orm::DB::statement("CREATE TABLE IF NOT EXISTS vertex_community_map(p_out "
                     "INTEGER NOT NULL,graph INTEGER NOT "
                     "NULL,vertex INTEGER NOT NULL,community INTEGER NOT NULL, "
                     "PRIMARY KEY(p_out, graph, "
                     "vertex))");
  Orm::DB::commit();

  Orm::DB::statement("CREATE TABLE IF NOT EXISTS edge_connection_map(p_out "
                     "INTEGER NOT NULL,graph INTEGER NOT "
                     "NULL,edge INTEGER NOT NULL, connection INTEGER NOT NULL, "
                     "PRIMARY KEY(p_out, graph, edge))");
  Orm::DB::commit();
}
std::string create_graph_tables_str() {
  std::string result = "CREATE TABLE IF NOT EXISTS edgelists(p_out INTEGER NOT "
                     "NULL,graph INTEGER NOT NULL,edge "
                     "INTEGER NOT NULL,\"from\" INTEGER NOT NULL,\"to\" INTEGER "
                     "NOT NULL, weight INTEGER NOT NULL, "
                     "PRIMARY KEY(p_out, graph, edge));\n";

  result+=
      "CREATE TABLE IF NOT EXISTS connection_community_map(p_out INTEGER NOT "
      "NULL,graph INTEGER "
      "NOT NULL,edge INTEGER NOT NULL,\"from\" INTEGER NOT NULL,\"to\" INTEGER NOT "
      "NULL, weight "
      "INTEGER NOT NULL, PRIMARY KEY(p_out, graph, edge));\n";

  result += "CREATE TABLE IF NOT EXISTS vertex_community_map(p_out "
                     "INTEGER NOT NULL,graph INTEGER NOT "
                     "NULL,vertex INTEGER NOT NULL,community INTEGER NOT NULL, "
                     "PRIMARY KEY(p_out, graph, "
                     "vertex));";

  result += "CREATE TABLE IF NOT EXISTS edge_connection_map(p_out "
                     "INTEGER NOT NULL,graph INTEGER NOT "
                     "NULL,edge INTEGER NOT NULL, connection INTEGER NOT NULL, "
                     "PRIMARY KEY(p_out, graph, edge));\n";
  return result;
}


void drop_graph_tables() {
  Orm::DB::statement("DROP TABLE IF EXISTS edgelists");
  Orm::DB::commit();
  Orm::DB::statement("DROP TABLE IF EXISTS connection_community_map");
  Orm::DB::commit();
  Orm::DB::statement("DROP TABLE IF EXISTS vertex_community_map");
  Orm::DB::commit();
  Orm::DB::statement("DROP TABLE IF EXISTS edge_connection_map");
  Orm::DB::commit();
}
std::string drop_graph_tables_str() {
  std::string result = "DROP TABLE IF EXISTS edgelists;\n";
  result += "DROP TABLE IF EXISTS connection_community_map;\n";
  result += "DROP TABLE IF EXISTS vertex_community_map;\n";
  result += "DROP TABLE IF EXISTS edge_connection_map;";
  return result;
}

void edge_upsert(const QString &table_name, uint32_t p_out, uint32_t graph,
                 const std::vector<uint32_t> &from,
                 const std::vector<uint32_t> &to,
                 std::vector<uint32_t> weight) {
  if (!weight.size())
  {
    weight = std::vector<uint32_t>(from.size(), 1);
  }
  QVector<QVector<QVariant>> rows(from.size());
  for (int i = 0; i < from.size(); i++) {
    Orm::DB::table("edge_connection_map")
      ->upsert({{{"p_out", p_out}, {"graph", graph}, {"edge", i}, {"from", from[i]}, {"to", to[i]}, {"weight", weight[i]}}}, {"p_out", "graph", "edge"}, {"from", "to", "weight"});
  }
}

void vcm_upsert(uint32_t p_out, uint32_t graph,
                const std::vector<uint32_t> &vcm) {
  // auto N_rows = std::min<uint32_t>({10, (uint32_t)vcm.size()});
  auto N_rows = vcm.size();
  QVector<QVector<QVariant>> rows(N_rows);
  for (int i = 0; i < N_rows; i++) {
    Orm::DB::table("edge_connection_map")
      ->upsert({{{"p_out", p_out}, {"graph", graph}, {"edge", i}, {"community", vcm[i]}}}, {"p_out", "graph", "edge"}, {"community"});
  }
}

void ecm_upsert(uint32_t p_out, uint32_t graph,
                const std::vector<uint32_t> &ecm) {
  QVector<Orm::WhereItem> row_inds;
  QVector<QVariantMap> row_datas;
  row_inds.reserve(ecm.size());
  row_datas.reserve(ecm.size());
  QVector<QVector<QVariant>> rows(ecm.size());
  for (int i = 0; i < ecm.size(); i++) {
    Orm::DB::table("edge_connection_map")
      ->upsert({{{"p_out", p_out}, {"graph", graph}, {"edge", i}, {"connection", ecm[i]}}}, {"p_out", "graph", "edge"}, {"connection"});
  }
}

std::vector<std::pair<uint32_t, uint32_t>> read_edgelist(uint32_t p_out,
                                                         uint32_t graph) {
  auto query = Orm::DB::table("edgelists")
                   // ->select({"from", "to"})
                   ->where({{"p_out", p_out}, {"graph", graph}})
                   .select({"from", "to"})
                   .get();
  std::vector<std::pair<uint32_t, uint32_t>> edgelist;
  auto q_size = query.size();
  edgelist.reserve(query.size());
  while (query.next()) {
    edgelist.push_back(std::make_pair(query.value("from").toUInt(),
                                      query.value("to").toUInt()));
  }
  return edgelist;
}

std::vector<std::array<uint32_t, 3>> ccm_read(uint32_t p_out, uint32_t graph) {
  auto query = Orm::DB::table("connection_community_map")
                   ->where({{"p_out", p_out}, {"graph", graph}})
                   .select({"from", "to", "weight"})
                   .get();
  std::vector<std::array<uint32_t, 3>> edgelist;
  edgelist.reserve(query.size());
  while (query.next()) {
    edgelist.push_back(std::array<uint32_t, 3>{query.value("from").toUInt(),
                                               query.value("to").toUInt(),
                                               query.value("to").toUInt()});
  }
  return edgelist;
}

std::vector<uint32_t> vcm_read(uint32_t p_out, uint32_t graph) {
  auto query = Orm::DB::table("vertex_community_map")->
                  //  ->where({{"p_out", p_out}, {"graph", graph}})
                  //  .orderBy("vertex", "asc")
                   select("community")
                   .get();
  std::vector<uint32_t> vcm;
  assert(query.size() > 0 &&"vcm_read: query.size() == 0");
  vcm.reserve(query.size());
  while (query.next()) {
    vcm.push_back(query.value("community").toUInt());
  }
  return vcm;
}

std::vector<uint32_t> ecm_read(uint32_t p_out, uint32_t graph) {
  auto query = Orm::DB::table("edge_connection_map")
                   ->where({{"p_out", p_out}, {"graph", graph}})
                   .orderBy("edge", "asc")
                   .select("connection")
                   .get();
  std::vector<uint32_t> ecm;
  assert(query.size() > 0 && "ecm_read: query.size() == 0");
  ecm.reserve(query.size());
  while (query.next()) {
    ecm.push_back(query.value("connection").toUInt());
  }
  return ecm;
}

void edgelist_insert(uint32_t p_out, uint32_t graph,
                     const std::vector<uint32_t> &from,
                     const std::vector<uint32_t> &to,
                     const std::vector<uint32_t> &weight) {
  edge_insert("connection_community_map", p_out, graph, from, to, weight);
}

void ccm_insert(uint32_t p_out, uint32_t graph,
                const std::vector<uint32_t> &from,
                const std::vector<uint32_t> &to,
                const std::vector<uint32_t> &weight) {
  edge_insert("edgelists", p_out, graph, from, to, weight);
}

} // namespace SBM_Graph
