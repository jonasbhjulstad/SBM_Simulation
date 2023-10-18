#ifndef SBM_DATABASE_DATABASE_SIMULATION_TABLES_HPP
#define SBM_DATABASE_DATABASE_SIMULATION_TABLES_HPP
#include <soci/soci.h>
namespace SBM_Database
{
void construct_sim_param_table(soci::session& sql, uint32_t Np);

void construct_simulation_tables(soci::session &sql, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt);

void drop_simulation_tables(soci::session &sql);

}// namespace SBM_Database
#endif
