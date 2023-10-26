#ifndef SBM_SIMULATION_SIMULATION_TABLES_HPP
#define SBM_SIMULATION_SIMULATION_TABLES_HPP
#include <Dataframe/Dataframe.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Types.hpp>
#include <orm/db.hpp>
#include <ranges>
namespace SBM_Database
{

    void create_simulation_tables()
    {
        Orm::DB::statement(
            "CREATE TABLE IF NOT EXISTS community_state(p_out INTEGER NOT NULL,graph INTEGER NOT "
            "NULL,simulation "
            "INTEGER NOT NULL,t INTEGER NOT NULL, community INTEGER NOT NULL, state INTEGER[3], "
            "PRIMARY KEY(p_out, graph, simulation, t, community))");

        Orm::DB::statement(
            "CREATE TABLE IF NOT EXISTS connection_events(p_out INTEGER NOT NULL,graph INTEGER NOT "
            "NULL,simulation "
            "INTEGER NOT NULL,t INTEGER NOT NULL, connection INTEGER NOT NULL, events INTEGER NOT "
            "NULL, "
            "PRIMARY KEY(p_out, graph, simulation, t, connection))");
        Orm::DB::statement(
            "CREATE TABLE IF NOT EXISTS infection_events(p_out INTEGER NOT NULL,graph INTEGER NOT "
            "NULL,simulation "
            "INTEGER NOT NULL,t INTEGER NOT NULL, connection INTEGER NOT NULL, events INTEGER NOT "
            "NULL, PRIMARY KEY(p_out, graph, simulation, t, connection))");

        auto p_I_table_create = [](auto postfix)
        {
            auto str = "CREATE TABLE IF NOT EXISTS p_Is_" + std::string(postfix) + "(p_out INTEGER NOT NULL,graph INTEGER NOT NULL,simulation "
                                                                                   "INTEGER NOT NULL,t INTEGER NOT NULL, connection INTEGER NOT NULL, p_I INTEGER NOT NULL, "
                                                                                   "PRIMARY KEY(p_out, graph, simulation, t, connection))";
            Orm::DB::statement(str.c_str());
        };

        for (auto &&control_type : {"Uniform", "Community"})
        {
            for (auto &&sim_type : {"Excitation", "Validation"})
            {
                p_I_table_create(control_type + std::string("_") + sim_type);
            }
        }

        Orm::DB::statement(
            "CREATE TABLE IF NOT EXISTS sim_params("
            "p_out_idx INTEGER NOT NULL,"
            "N_pop INTEGER NOT NULL,"
            "p_in_value REAL NOT NULL,"
            "p_out_value REAL NOT NULL,"
            "N_graphs INTEGER NOT NULL,"
            "N_sims INTEGER NOT NULL,"
            "Nt INTEGER NOT NULL,"
            "Nt_alloc INTEGER NOT NULL,"
            "seed INTEGER NOT NULL,"
            "p_I_min REAL NOT NULL,"
            "p_I_max REAL NOT NULL,"
            "p_out_idx INTEGER NOT NULL,"
            "p_R REAL NOT NULL,"
            "p_I0 REAL NOT NULL,"
            "p_R0 REAL NOT NULL,"
            "PRIMARY KEY (p_out)");
        Orm::DB::statement(
            "CREATE TABLE IF NOT EXISTS N_communities("
            "p_out_idx INTEGER NOT NULL,"
            "graph INTEGER NOT NULL,"
            "simulation INTEGER NOT NULL,"
            "communities INTEGER NOT NULL,"
            "PRIMARY KEY(p_out, graph, simulation))");
    }

    void community_state_insert(uint32_t p_out, uint32_t graph, Dataframe::Dataframe_t<State_t, 4> &df)
    {
        auto N_sims = df.size();
        auto Nt = df[0].size();
        auto N_communities = df[0][0].size();
        QVector<Orm::WhereItem> row_inds;
        QVector<QVariantMap> row_datas;
        uint32_t N_rows = N_sims * Nt * N_communities;
        row_inds.reserve(N_rows);
        row_datas.reserve(N_rows);
        for (const auto &[sim_id, sim_df] : ranges::views::enumerate(df.data))
        {
            for (const auto &[t, t_df] : ranges::views::enumerate(sim_df.data))
            {
                for (const auto &[community, state] : ranges::views::enumerate(t_df.data))
                {
                    row_inds.push_back({{"p_out", p_out}, {"graph", graph}, {"simulation", sim_id}, {"t", t}});
                row_datas.push_back({"community", QVariant::fromValue(QVector({community})});
                }
            }
        }
        Orm::DB::table("community_state")->insert(QVector<QString>{"p_out", "graph", "simulation", "t", "community"}, rows);
    }

    template <typename T = uint32_t>
    void connection_insert(const QString &table_name, uint32_t p_out, uint32_t graph, Dataframe::Dataframe_t<T, 4> &df)
    {
        auto N_sims = df.size();
        auto Nt = df[0].size();
        auto N_communities = df[0][0].size();
        QVector<Orm::WhereItem> row_inds;
        QVector<QVariantMap> row_datas;
        uint32_t N_rows = N_sims * Nt * N_communities;
        row_inds.reserve(N_rows);
        row_datas.reserve(N_rows);
        for (const auto &[sim_id, sim_df] : ranges::views::enumerate(df.data))
        {
            for (const auto &[t, t_df] : ranges::views::enumerate(sim_df.data))
            {
                for (const auto &[community, state] : ranges::views::enumerate(t_df.data))
                {
                    row_inds.push_back({{"p_out", p_out}, {"graph", graph}, {"simulation", sim_id}, {"t", t}});
                row_datas.push_back({"community", QVariant::from_value(QVector({community})});
                }
            }
        }
        Orm::DB::table(table_name)->insert(QVector<QString>{"p_out", "graph", "simulation", "t", "community"}, rows);
    }
    void sim_param_insert(const QJsonDocument &sim_param)
    {
        // convert QJsonDocument to QVariant
        QVariantMap sim_param_map = sim_param.toVariant().toMap();
        QVector<QVariant> N_communities = sim_param_map["N_communities"].toList().toVector();
        auto p_out_idx = sim_param_map["p_out_idx"].toInt();
        // drop N_communities from sim_param_map
        sim_param_map.remove("N_communities");
        // insert sim_param_map into sim_params table
        Orm::DB::table("sim_params")->insert(sim_param_map);

        Orm::DB::table("N_communities")->insert({"p_out_idx", p_out_idx}, N_communities);
    }

    QJsonDocument sim_param_read(uint32_t p_out)
    {
        auto sim_param = Orm::DB::table("sim_params")->where("p_out_idx", p_out)->first();
        auto N_communities = Orm::DB::table("N_communities")->where("p_out_idx", p_out)->get();
        sim_param["N_communities"] = N_communities;
        return sim_param;
    }

    void drop_simulation_tables()
    {
        Orm::DB::statement("DROP TABLE IF EXISTS community_state");
        Orm::DB::statement("DROP TABLE IF EXISTS connection_events");
        Orm::DB::statement("DROP TABLE IF EXISTS infection_events");
        auto p_I_table_drop = [](auto postfix)
        { Orm::DB::statement(("DROP TABLE IF EXISTS p_Is_" + std::string(postfix)).c_str()); };
        for (auto &&control_type : {"Uniform", "Community"})
        {
            for (auto &&sim_type : {"Excitation", "Validation"})
            {
                p_I_table_drop(control_type + std::string("_") + sim_type);
            }
        }

        Orm::DB::statement("DROP TABLE IF EXISTS sim_params");
        Orm::DB::statement("DROP TABLE IF EXISTS N_communities");
    }
} // namespace SBM_Database

#endif
