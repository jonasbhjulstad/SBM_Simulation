#ifndef SBM_DATABASE_GRAPH_TABLES_HPP
#define SBM_DATABASE_GRAPH_TABLES_HPP
#include <SBM_Graph/Common.hpp>
namespace SBM_Graph
{
  void create_graph_tables()
  {
    Orm::DB::statement(
        "CREATE TABLE IF NOT EXISTS edgelists(p_out INTEGER NOT NULL,graph INTEGER NOT NULL,edge "
        "INTEGER NOT NULL,'from' INTEGER NOT NULL,'to' INTEGER NOT NULL, weight INTEGER NOT NULL, "
        "PRIMARY KEY(p_out, graph, edge))");
    Orm::DB::statement(
        "CREATE TABLE IF NOT EXISTS connection_community_map(p_out INTEGER NOT NULL,graph INTEGER "
        "NOT NULL,edge INTEGER NOT NULL,'from' INTEGER NOT NULL,'to' INTEGER NOT NULL, weight "
        "INTEGER NOT NULL, PRIMARY KEY(p_out, graph, edge))");
    Orm::DB::statement(
        "CREATE TABLE IF NOT EXISTS vertex_community_map(p_out INTEGER NOT NULL,graph INTEGER NOT "
        "NULL,vertex INTEGER NOT NULL,community INTEGER NOT NULL, PRIMARY KEY(p_out, graph, "
        "vertex))");
    Orm::DB::statement(
        "CREATE TABLE IF NOT EXISTS edge_community_map(p_out INTEGER NOT NULL,graph INTEGER NOT "
        "NULL,edge INTEGER NOT NULL, community INTEGER NOT NULL, PRIMARY KEY(p_out, graph, edge))");
  }

  void drop_graph_tables()
  {
    Orm::DB::statement("DROP TABLE IF EXISTS edgelists");
    Orm::DB::statement("DROP TABLE IF EXISTS connection_community_map");
    Orm::DB::statement("DROP TABLE IF EXISTS vertex_community_map");
    Orm::DB::statement("DROP TABLE IF EXISTS edge_community_map");
  }
  void edge_insert(const QString &table_name, uint32_t p_out, uint32_t graph,
                   const std::vector<uint32_t> &from, const std::vector<uint32_t> &to,
                   const std::vector<uint32_t> &weight = {})
  {
    QVector<Orm::WhereItem> row_inds;
    QVector<QVariantMap> row_datas;
    row_inds.reserve(from.size());
    row_datas.reserve(from.size());
    QVector<QVector<QVariant>> rows(from.size());
    for (int i = 0; i < from.size(); i++)
    {
      rows[i] = {p_out, graph, i, from[i], to[i], (weight.size()) ? weight[i] : 1};
    }
    Orm::DB::table(table_name)
        ->insert(QVector<QString>{"p_out", "graph", "edge", "from", "to", "weight"}, rows);
  }
  std::vector<std::pair<uint32_t, uint32_t>> read_edgelist(uint32_t p_out, uint32_t graph)
  {
    auto query = Orm::DB::table("edgelists")
                     // ->select({"from", "to"})
                     ->where({{"p_out", p_out}, {"graph", graph}})
                     .select({"from", "to"})
                     .get();
    std::vector<std::pair<uint32_t, uint32_t>> edgelist;
    edgelist.reserve(query.size());
    while (query.next())
    {
      edgelist.push_back(std::make_pair(query.value("from").toUInt(), query.value("to").toUInt()));
    }
    return edgelist;
  }

  std::vector<std::array<uint32_t, 3>> read_ccm(uint32_t p_out, uint32_t graph)
  {
    auto query = Orm::DB::table("connection_community_map")
                     ->where({{"p_out", p_out}, {"graph", graph}})
                     .select({"from", "to", "weight"})
                     .get();
    std::vector<std::array<uint32_t, 3>> edgelist;
    edgelist.reserve(query.size());
    while (query.next())
    {
      edgelist.push_back(std::array<uint32_t, 3>{query.value("from").toUInt(), query.value("to").toUInt(), query.value("to").toUInt()});
    }
    return edgelist;
  }

  std::vector<uint32_t> read_vcm(uint32_t p_out, uint32_t graph)
  {
    auto query = Orm::DB::table("vertex_community_map")
                     ->where({{"p_out", p_out}, {"graph", graph}})
                     .orderBy("vertex", "asc")
                     .select("community")
                     .get();
    std::vector<uint32_t> vcm;
    vcm.reserve(query.size());
    while (query.next())
    {
      vcm.push_back(query.value("community").toUInt());
    }
    return vcm;
  }

  std::vector<uint32_t> read_ecm(uint32_t p_out, uint32_t graph)
  {
    auto query = Orm::DB::table("edge_community_map")
                     ->where({{"p_out", p_out}, {"graph", graph}})
                     .orderBy("edge", "asc")
                     .select("community")
                     .get();
    std::vector<uint32_t> ecm;
    ecm.reserve(query.size());
    while (query.next())
    {
      ecm.push_back(query.value("community").toUInt());
    }
    return ecm;
  }

  void edgelist_insert(uint32_t p_out, uint32_t graph, const std::vector<uint32_t> &from,
                       const std::vector<uint32_t> &to, const std::vector<uint32_t> &weight = {})
  {
    edge_insert("connection_community_map", p_out, graph, from, to, weight);
  }

  void ccm_insert(uint32_t p_out, uint32_t graph, const std::vector<uint32_t> &from,
                  const std::vector<uint32_t> &to, const std::vector<uint32_t> &weight = {})
  {
    edge_insert("edgelists", p_out, graph, from, to, weight);
  }

} // namespace SBM_Graph

#endif
