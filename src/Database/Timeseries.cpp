#include <Sycl_Graph/Database/Timeseries.hpp>
#include <iostream>

void create_timeseries_table(pqxx::connection &con, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt, const std::string& table_name)
{
  auto max_constraint = [](std::string idx_name, auto N_max) {return "CONSTRAINT max_index_" + idx_name + " CHECK (" + idx_name + " < " + std::to_string(N_max) + ")";};

  auto work = pqxx::work(con);
  std::string command = "CREATE TABLE IF NOT EXISTS " + table_name + " ("
            "  p_out INTEGER NOT NULL " + max_constraint("p_out", Np) + ","
            "  graph INTEGER NOT NULL " + max_constraint("graph", Ng) + ","
            "  sim INTEGER NOT NULL " + max_constraint("sim", Ns) + ","
            "  time INTEGER NOT NULL " + max_constraint("time", Nt) + ","
            "  state real[] NOT NULL,"
            "  PRIMARY KEY (p_out, graph, sim, time)"
            ")";
  work.exec(command.c_str());
  work.commit();
}


// void create_timeseries_table(pqxx::connection &con, uint32_t N_cols)
// {
//   auto work = pqxx::work(con);
//   work.exec(("CREATE TABLE IF NOT EXISTS timeseries ("
//              "  p_out INTEGER NOT NULL,"
//              "  graph INTEGER NOT NULL,"
//              "  time INTEGER NOT NULL,"
//              "  state real[" +
//              std::to_string(N_cols) + "] NOT NULL,"
//                                       "  PRIMARY KEY (p_out, graph, time)"
//                                       ")")
//                 .c_str());
//   work.commit();
// }


Eigen::MatrixXf get_timestep(pqxx::connection &con, int graph_id, float p_out, uint32_t t, uint32_t N_cols)
{
  auto work = pqxx::work(con);

  auto result = work.exec("SELECT state FROM timeseries WHERE p_out = " + std::to_string(p_out) + " AND graph = " + std::to_string(graph_id) + " AND time = " + std::to_string(t));

  Eigen::MatrixXf timeseries(result.size(), N_cols);

  auto result_idx = result[0];
  for (int i = 0; i < result.size(); i++)
  {
    auto res = result[i]["state"].as_array();
    auto entry = res.get_next();
    for (int j = 0; j < N_cols; j++)
    {
      if (!is_string_entry(entry))
      {
        throw std::runtime_error("Expected more columns database timeseries, got to index: " + std::to_string(j) + " of " + std::to_string(N_cols));
      }
      entry = res.get_next();
      timeseries(i, j) = std::stoi(std::get<1>(entry));
    }
  }

  return timeseries;
}

void set_timeseries(pqxx::work& work, uint32_t p_out, uint32_t graph, uint32_t Np, uint32_t Ng, const std::string& table_name, Eigen::MatrixXf &timeseries)
{
  auto Nt = timeseries.rows();
  auto cols = timeseries.cols();
  for (int t = 0; t < Nt; t++)
  {
    std::string state = "{";
    for (int i = 0; i < cols; i++)
    {
      state += std::to_string(timeseries(t, i));
      if (i < cols - 1)
      {
        state += ",";
      }
    }
    state += "}";
    work.exec("INSERT INTO " + table_name + " (p_out, graph, time, state) VALUES (" + std::to_string(p_out) + ", " + std::to_string(graph) + ", " + std::to_string(t) + ", " + state + ")");
  }
}

void set_timeseries(pqxx::connection &con, uint32_t p_out, uint32_t graph, uint32_t Np, uint32_t Ng, const std::string& table_name, Eigen::MatrixXf &timeseries)
{
  auto work = pqxx::work(con);
  set_timeseries(work, p_out, graph, Np, Ng, table_name, timeseries);
  work.commit();
}


void print_timestep(pqxx::connection &con, int graph_id, float p_out, uint32_t t, uint32_t N_cols)
{
  auto ts = get_timestep(con, graph_id, p_out, t, N_cols);
  std::cout << ts << std::endl;
}

Eigen::MatrixXf get_timeseries(pqxx::connection &con, int graph_id, float p_out, uint32_t N_cols)
{
  auto work = pqxx::work(con);

  auto result = work.exec("SELECT state FROM timeseries WHERE p_out = " + std::to_string(p_out) + " AND graph = " + std::to_string(graph_id));
  Eigen::MatrixXf timeseries(result.size(), N_cols);

  auto result_idx = result[0];
  for (int i = 0; i < result.size(); i++)
  {
    auto res = result[i]["state"].as_array();
    auto entry = res.get_next();
    for (int j = 0; j < N_cols; j++)
    {
      entry = res.get_next();
      if (!is_string_entry(entry))
      {
        throw std::runtime_error("Expected more columns database timeseries, got to index: " + std::to_string(j) + " of " + std::to_string(N_cols));
      }
      timeseries(i, j) = std::stoi(std::get<1>(entry));
    }
  }
  return timeseries;
}

void print_timeseries(pqxx::connection &con, int graph_id, float p_out, uint32_t N_cols)
{
  auto ts = get_timeseries(con, graph_id, p_out, N_cols);
  std::cout << ts << std::endl;
}

void delete_table(pqxx::connection &con, const std::string &table_name)
{
  auto work = pqxx::work(con);
  work.exec("DROP TABLE IF EXISTS " + table_name);
  work.commit();
}

void drop_table(pqxx::connection &con, const std::string& table_name)
{
  auto work = pqxx::work(con);
  work.exec(("DROP TABLE IF EXISTS " + table_name).c_str());
  work.commit();
}
