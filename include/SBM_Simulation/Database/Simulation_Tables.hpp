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
                            uint32_t t_offset = 0);

template <typename T>
Dataframe::Dataframe_t<T, 3>
connection_read(const QString &table_name, uint32_t p_out, uint32_t graph, uint32_t N_sims, uint32_t Nt, uint32_t N_cols, const QString& colname);
template <>
Dataframe::Dataframe_t<uint32_t, 3>
connection_read<uint32_t>(const QString &table_name, uint32_t p_out,
                          uint32_t graph, uint32_t N_sims, uint32_t Nt,
                          uint32_t N_cols, const QString &colname);
template <>
Dataframe::Dataframe_t<float, 3>
connection_read<float>(const QString &table_name, uint32_t p_out,
                       uint32_t graph, uint32_t N_sims, uint32_t Nt,
                       uint32_t N_cols, const QString &colname);

void sim_param_upsert(const QJsonObject &sim_param); 

Sim_Param sim_param_read(uint32_t p_out);

} // namespace SBM_Database

#endif
