#ifndef SBM_SIMULATION_SIMULATION_TABLES_HPP
#define SBM_SIMULATION_SIMULATION_TABLES_HPP
#include <Dataframe/Dataframe.hpp>
#include <SBM_Simulation/Types/SIR_Types.hpp>
#include <orm/db.hpp>
#include <ranges>
namespace SBM_Database {


void community_state_insert(uint32_t p_out, uint32_t graph,
                            Dataframe::Dataframe_t<State_t, 3> &df,
                            uint32_t t_offset = 0) {
  auto N_sims = df.size();
  auto Nt = df[0].size();
  auto N_communities = df[0][0].size();
  uint32_t N_rows = N_sims * Nt * N_communities;
  QVector<QVector<QVariant>> rows(N_rows);

  for (int sim_id = 0; sim_id < N_sims; sim_id++) {
    for (int t = 0; t < Nt; t++) {
      for (int community = 0; community < N_communities; community++) {
        auto row_ind = sim_id * Nt * N_communities + t * N_communities +
                       community + t_offset * N_communities;
        rows[row_ind] = {p_out,graph,sim_id,t, community};
      }
    }
  }
  Orm::DB::table("community_state")
      ->insert(
          QVector<QString>{"p_out", "graph", "simulation", "t", "community"},
          rows);
}

template <typename T = uint32_t>
void connection_insert(const QString &table_name, uint32_t p_out,
                       uint32_t graph, Dataframe::Dataframe_t<T, 3> &df,
                       uint32_t t_offset = 0) {
  auto N_sims = df.size();
  auto Nt = df[0].size();
  auto N_communities = df[0][0].size();


  uint32_t N_rows = N_sims * Nt * N_communities;
  QVector<QVector<QVariant>> rows(N_rows);

  for (int sim_id = 0; sim_id < N_sims; sim_id++) {
    for (int t = 0; t < Nt; t++) {
      for (int community = 0; community < N_communities; community++) {
                auto row_ind = sim_id * Nt * N_communities + t * N_communities +
                       community + t_offset * N_communities;
                auto community = df[sim_id][t][community];
                rows[row_ind] = {{p_out, graph, sim_id, t, community}};
      }
    }
  }
  Orm::DB::table(table_name)
      ->insert(
          QVector<QString>{"p_out", "graph", "simulation", "t", "community"},
          rows);
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

void drop_simulation_tables() {
  Orm::DB::statement("DROP TABLE IF EXISTS community_state");
  Orm::DB::statement("DROP TABLE IF EXISTS connection_events");
  Orm::DB::statement("DROP TABLE IF EXISTS infection_events");
  auto p_I_table_drop = [](auto postfix) {
    Orm::DB::statement(
        ("DROP TABLE IF EXISTS p_Is_" + std::string(postfix)).c_str());
  };
  for (auto &&control_type : {"Uniform", "Community"}) {
    for (auto &&sim_type : {"Excitation", "Validation"}) {
      p_I_table_drop(control_type + std::string("_") + sim_type);
    }
  }

  Orm::DB::statement("DROP TABLE IF EXISTS simulation_parameters");
  Orm::DB::statement("DROP TABLE IF EXISTS N_communities");
}
} // namespace SBM_Database

#endif
