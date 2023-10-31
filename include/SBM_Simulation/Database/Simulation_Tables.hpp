#ifndef SBM_SIMULATION_SIMULATION_TABLES_HPP
#define SBM_SIMULATION_SIMULATION_TABLES_HPP
#include <Dataframe/Dataframe.hpp>
#include <SBM_Simulation/Types/SIR_Types.hpp>
#include <orm/db.hpp>
#include <ranges>
namespace SBM_Database {

template <typename T = uint32_t>
void connection_upsert(const QString &table_name, uint32_t p_out,
                       uint32_t graph, Dataframe::Dataframe_t<T, 3> &df,
                       uint32_t t_offset = 0) {
  auto N_sims = df.size();
  auto Nt = df[0].size();
  auto N_communities = df[0][0].size();

  uint32_t N_rows = N_sims * Nt * N_communities;
  QVector<QVector<QVariant>> rows(N_rows);

  for (int sim_id = 0; sim_id < N_sims; sim_id++) {
    for (int t = 0; t < Nt; t++) {
      for (int connection = 0; connection < N_communities; connection++) {
        auto row_ind = sim_id * Nt * N_communities + t * N_communities +
                       connection + t_offset * N_communities;
        auto connection_value = df[sim_id][t][connection];
        Orm::DB::table(table_name)
            ->upsert({{{"p_out", p_out},
                       {"graph", graph},
                       {"simulation", sim_id},
                       {"t", t},
                       {"connection", connection},
                       {"value", connection_value}}},
                     {"p_out", "graph", "simulation", "t", "community"},
                     {"value"});
      }
    }
  }
}

void community_state_upsert(uint32_t p_out, uint32_t graph,
                            Dataframe::Dataframe_t<State_t, 3> &df,
                            uint32_t t_offset = 0) {
  auto N_sims = df.size();
  auto Nt = df[0].size();
  auto N_communities = df[0][0].size();
  uint32_t N_rows = N_sims * Nt * N_communities;

  for (int sim_id = 0; sim_id < N_sims; sim_id++) {
    for (int t = 0; t < Nt; t++) {
      for (int community = 0; community < N_communities; community++) {
        Orm::DB::table("community_state")
            ->upsert({{{"p_out", p_out},
                       {"graph", graph},
                       {"simulation", sim_id},
                       {"t", t},
                       {"community", community},
                       {"S", df[sim_id][t][community][0]},
                       {"I", df[sim_id][t][community][1]},
                       {"R", df[sim_id][t][community][2]}}},
                     {"p_out", "graph", "simulation", "t", "community"},
                     {"S", "I", "R"});
      }
    }
  }
}

template <typename T>
Dataframe::Dataframe_t<uint32_t, 3>
connection_read(const QString &table_name, uint32_t p_out, uint32_t graph,
                uint32_t N_sims, uint32_t Nt, uint32_t N_cols);
template <>
Dataframe::Dataframe_t<uint32_t, 3>
connection_read(const QString &table_name, uint32_t p_out, uint32_t graph,
                uint32_t N_sims, uint32_t Nt, uint32_t N_cols) {
  QVector<QString> columns_qt;
  columns_qt.reserve(columns.size());
  for (const auto &col : columns) {
    columns_qt.push_back(col.c_str());
  }
  auto query = Orm::DB::table(table_name)
                   ->where({{"p_out", p_out}, {"graph", graph}})
                   .select(columns_qt);
  Dataframe::Dataframe_t<uint32_t, 3> result({N_sims, Nt, N_cols});
  uint32_t sim = 0;
  uint32_t t = 0;
  uint32_t col = 0;
  QVector<QVariant> row;
  while (query.next()) {
    row = query.value(0).toList();
    result[sim][t][col] = row[0].toInt();
    col++;
    if (col == N_cols) {
      col = 0;
      t++;
      if (t == Nt) {
        t = 0;
        sim++;
      }
    }
  }

  return result;
}

template <>
Dataframe::Dataframe_t<float, 3>
connection_read(const QString &table_name, uint32_t p_out, uint32_t graph,
                uint32_t N_sims, uint32_t Nt, uint32_t N_cols) {
  QVector<QString> columns_qt;
  columns_qt.reserve(columns.size());
  for (const auto &col : columns) {
    columns_qt.push_back(col.c_str());
  }
  auto query = Orm::DB::table(table_name)
                   ->where({{"p_out", p_out}, {"graph", graph}})
                   .select(columns_qt);
  Dataframe::Dataframe_t<uint32_t, 3> result({N_sims, Nt, N_cols});
  uint32_t sim = 0;
  uint32_t t = 0;
  uint32_t col = 0;
  QVector<QVariant> row;
  while (query.next()) {
    row = query.value(0).toList();
    result[sim][t][col] = row[0].toDouble();
    col++;
    if (col == N_cols) {
      col = 0;
      t++;
      if (t == Nt) {
        t = 0;
        sim++;
      }
    }
  }

  return result;
}

void sim_param_insert(const QJsonDocument &sim_param) {
  QVariantMap sim_param_map = sim_param.toVariant().toMap();
  auto p_out_id = sim_param_map["p_out_id"].toInt();
  Orm::DB::table("simulation_parameters")->insert(sim_param_map);
}

Sim_Param sim_param_read(uint32_t p_out) {
  auto sim_param_json = Orm::DB::table("simulation_parameters")
                            ->whereEq("p_out_id", p_out)
                            ->first();

  // auto N_communities = Orm::DB::table("N_communities")->whereEq("p_out_id",
  // p_out)->get();
  return Sim_Param::from_json(sim_param_json);
}

} // namespace SBM_Database

#endif
