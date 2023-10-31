#ifndef SBM_DATABASE_GRAPH_TABLES_HPP
#define SBM_DATABASE_GRAPH_TABLES_HPP
#include <SBM_Graph/Common.hpp>
#include <orm/db.hpp>
// #include <QVariant>
namespace SBM_Graph
{
  void create_graph_tables();

  void drop_graph_tables();

  std::string create_graph_tables_str();

  std::string drop_graph_tables_str();

  void edge_insert(const QString &table_name, uint32_t p_out, uint32_t graph,
                   const std::vector<uint32_t> &from,
                   const std::vector<uint32_t> &to,
                   std::vector<uint32_t> weight = {});

  void vcm_insert(uint32_t p_out, uint32_t graph,
                  const std::vector<uint32_t> &vcm);

  void ecm_insert(uint32_t p_out, uint32_t graph,
                  const std::vector<uint32_t> &ecm);

  std::vector<std::pair<uint32_t, uint32_t>> read_edgelist(uint32_t p_out,
                                                           uint32_t graph);

  std::vector<std::array<uint32_t, 3>> ccm_read(uint32_t p_out, uint32_t graph);

  std::vector<uint32_t> vcm_read(uint32_t p_out, uint32_t graph);

  std::vector<uint32_t> ecm_read(uint32_t p_out, uint32_t graph);

  void edgelist_insert(uint32_t p_out, uint32_t graph,
                       const std::vector<uint32_t> &from,
                       const std::vector<uint32_t> &to,
                       const std::vector<uint32_t> &weight = {});

  void ccm_insert(uint32_t p_out, uint32_t graph,
                  const std::vector<uint32_t> &from,
                  const std::vector<uint32_t> &to,
                  const std::vector<uint32_t> &weight = {});

} // namespace SBM_Graph

#endif
