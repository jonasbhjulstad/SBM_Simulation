#ifndef SYCL_GRAPH_DATABASE_DATAFRAME_HPP
#define SYCL_GRAPH_DATABASE_DATAFRAME_HPP
#include <Sycl_Graph/Database/Timeseries.hpp>
#include <Sycl_Graph/Dataframe/Dataframe.hpp>
#include <Sycl_Graph/SIR_Types.hpp>

void create_table(pqxx::connection &con, const std::string &table_name,
                             const std::vector<std::string> &indices,
                             const std::vector<std::string> &data_names,
                             std::vector<std::string> data_types = {});

template <typename T>
void write_timeseries(pqxx::work &work, uint32_t p_out, uint32_t graph, uint32_t sim, Dataframe_t<T, 2> &timeseries, const std::string &table_name, uint32_t offset = 0)
{
    auto Nt = timeseries.size();
    auto cols = timeseries[0].size();
    for (int t = offset; t < offset + Nt; t++)
    {
        std::stringstream ss;
        ss << "'{ ";
        for (int i = 0; i < cols; i++)
        {
            if (i < cols - 1)
            {
                ss << timeseries(t, i) << ",";
            }
            else
            {
                ss << timeseries(t, i);
            }
        }
        ss << " }'";
        work.exec("INSERT INTO " + table_name + " (p_out, graph, sim, time, state) VALUES (" + std::to_string(p_out) + ", " + std::to_string(graph) + ", " + std::to_string(sim) + ", " + std::to_string(t) + ", " + ss.str() + ")"
                                                                                                                                                                                                                                " ON CONFLICT ON CONSTRAINT " +
                  table_name + "_pkey DO UPDATE SET state = " +
                  ss.str() + ";");
    }
}
template <typename T>
void write_simseries(pqxx::work &work, uint32_t p_out, uint32_t graph, Dataframe_t<T, 3> &simseries, const std::string &table_name, uint32_t offset = 0)
{
    auto N_sims = simseries.size();
    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        write_timeseries(work, p_out, graph, sim_idx, simseries[sim_idx], table_name, offset);
    }
}
template <typename T>
void write_graphseries(pqxx::work &work, uint32_t p_out, Dataframe_t<T, 4> &graphseries, const std::string &table_name, uint32_t offset = 0)
{
    auto [N_graphs, N_sims, Nt, N_cols] = graphseries.get_ranges();
    for (int graph_idx = 0; graph_idx < N_graphs; graph_idx++)
    {
        write_simseries(work, p_out, graph_idx, graphseries[graph_idx], table_name, offset);
    }
}

template <typename T>
void write_timeseries(pqxx::connection &con, uint32_t p_out, uint32_t graph, uint32_t sim, Dataframe_t<T, 2> &timeseries, const std::string &table_name, uint32_t offset = 0)
{
    auto work = pqxx::work(con);
    write_timeseries(work, p_out, graph, sim, timeseries, table_name, offset);
    work.commit();
}

template <typename T>
void write_simseries(pqxx::connection &con, uint32_t p_out, uint32_t graph, Dataframe_t<T, 3> &simseries, const std::string &table_name, uint32_t offset = 0)
{
    auto work = pqxx::work(con);
    write_simseries(work, p_out, graph, simseries, table_name, offset);
    work.commit();
}

template <typename T>
void write_graphseries(pqxx::connection &con, uint32_t p_out, Dataframe_t<T, 4> &graphseries, const std::string &table_name, uint32_t offset = 0)
{
    auto work = pqxx::work(con);
    write_graphseries(work, p_out, graphseries, table_name, offset);
    work.commit();
}

Dataframe_t<State_t, 2> read_timeseries(pqxx::connection &con, int p_out, int sim_id, int graph_id, uint32_t N_cols);

Dataframe_t<State_t, 3> read_simseries(pqxx::connection &con, int p_out, int graph_id, const std::vector<uint32_t> &sim_ids, uint32_t N_cols);

Dataframe_t<State_t, 4> read_graphseries(pqxx::connection &con, int p_out, const std::vector<uint32_t> &graph_ids, uint32_t N_cols);

#endif
