#ifndef SBM_SIMULATION_SIMULATION_TABLES_HPP
#define SBM_SIMULATION_SIMULATION_TABLES_HPP
#include <Dataframe/Dataframe.hpp>
#include <SBM_Simulation/Types/SIR_Types.hpp>
#include <SBM_Simulation/Types/Sim_Types.hpp>
#include <orm/db.hpp>
#include <ranges>
#include <QJsonObject>
#include <QString>
#include <QVector>
namespace SBM_Simulation {

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
Dataframe::Dataframe_t<T, 3>
connection_read(const QString &table_name, uint32_t p_out, uint32_t graph, uint32_t N_sims, uint32_t Nt, uint32_t N_cols, const QString& colname);


template <>
Dataframe::Dataframe_t<uint32_t, 3>
connection_read<uint32_t>(const QString &table_name, uint32_t p_out, uint32_t graph, uint32_t N_sims, uint32_t Nt, uint32_t N_cols, const QString& colname){
  auto query = Orm::DB::table(table_name)
                   ->where({{"p_out", p_out}, {"graph", graph}})
                   .get();
  Dataframe::Dataframe_t<uint32_t, 3> result(std::array<uint32_t, 3>({N_sims, Nt, N_cols}));
  uint32_t sim = 0;
  uint32_t t = 0;
  uint32_t col = 0;
  QVector<QVariant> row;
  while (query.next()) {
    auto sim_id = query.value("simulation").toInt();
    auto t = query.value("t").toInt();
    auto col_id = query.value(colname).toInt();
    result[sim_id][t][col_id] = query.value("value").toInt();
  }
  return result;
}

template <>
Dataframe::Dataframe_t<float, 3>
connection_read<float>(const QString &table_name, uint32_t p_out, uint32_t graph, uint32_t N_sims, uint32_t Nt, uint32_t N_cols, const QString& colname) {
  auto query = Orm::DB::table(table_name)
                   ->where({{"p_out", p_out}, {"graph", graph}})
                   .get();
  Dataframe::Dataframe_t<float, 3> result(std::array<uint32_t, 3>({N_sims, Nt, N_cols}));
  uint32_t sim = 0;
  uint32_t t = 0;
  uint32_t col = 0;
  QVector<QVariant> row;
  while (query.next()) {
    auto sim_id = query.value("simulation").toInt();
    auto t = query.value("t").toInt();
    auto col_id = query.value(colname).toInt();
    result[sim_id][t][col_id] = query.value("value").toFloat();
  }
  return result;
}

void sim_param_upsert(const QJsonObject &sim_param) {
  Orm::DB::table("simulation_parameters")->upsert({sim_param.toVariantMap()}, {"p_out"}, sim_param.keys());
}

Sim_Param sim_param_read(uint32_t p_out) {
  auto query = Orm::DB::table("simulation_parameters")
                            ->whereEq("p_out_id", p_out).first();
  //convert to json
  auto to_float = [](auto num) { return num.toFloat(); };
  auto to_uint = [](auto num) { return num.toUInt(); };

  Sim_Param result{to_uint(query.value(0)),
                   to_uint(query.value(1)),
                   to_uint(query.value(2)),
                   to_uint(query.value(3)),
                   to_uint(query.value(4)),
                   to_uint(query.value(5)),
                   to_uint(query.value(6)),
                   to_uint(query.value(7)),
                   to_uint(query.value(8)),
                   to_float(query.value(9)),
                   to_float(query.value(10)),
                   to_float(query.value(11)),
                   to_float(query.value(12)),
                   to_float(query.value(13)),
                   to_float(query.value(14)),
                   to_float(query.value(15))};

  // auto N_communities = Orm::DB::table("N_communities")->whereEq("p_out_id",
  // p_out)->get();
  return result;
}

} // namespace SBM_Database

#endif
