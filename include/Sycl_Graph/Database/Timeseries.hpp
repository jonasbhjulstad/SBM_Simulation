#ifndef SYCL_GRAPH_DATABASE_TIMESERIES_HPP
#define SYCL_GRAPH_DATABASE_TIMESERIES_HPP
#include <Eigen/Dense>
#include <pqxx/pqxx>
#include <Sycl_Graph/Database/Table.hpp>

void create_timeseries_table(pqxx::connection &con);

void create_timeseries_table(pqxx::connection &con, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt, const std::string& table_name);

Eigen::MatrixXf get_timestep(pqxx::connection &con, int graph_id, float p_out, uint32_t t, uint32_t N_cols);

void set_timeseries(pqxx::connection &con, uint32_t p_out, uint32_t graph, uint32_t Np, uint32_t Ng, Eigen::MatrixXf &timeseries);

void print_timestep(pqxx::connection &con, int graph_id, float p_out, uint32_t t, uint32_t N_cols);

Eigen::MatrixXf get_timeseries(pqxx::connection &con, int graph_id, float p_out, uint32_t N_cols);

void print_timeseries(pqxx::connection &con, int graph_id, float p_out, uint32_t N_cols);

void delete_table(pqxx::connection &con, const std::string &table_name);

void drop_table(pqxx::connection &con, const std::string& table_name);


#endif // TIMESERIES_HPP
