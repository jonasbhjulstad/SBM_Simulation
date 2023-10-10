#ifndef SYCL_GRAPH_DATABASE_SIMULATION_TABLES_HPP
#define SYCL_GRAPH_DATABASE_SIMULATION_TABLES_HPP
#include <pqxx/pqxx>
void construct_simulation_tables(pqxx::connection &con, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt);
void drop_simulation_tables(pqxx::connection &con);

#endif
