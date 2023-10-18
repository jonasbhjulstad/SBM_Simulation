#include <SBM_Simulation/Simulation/Sim_Types.hpp>

using namespace SBM_Database;
int main()
{
    pqxx::connection con("dbname=postgres user=postgres password=postgres");
    drop_simulation_tables(sql);
    construct_simulation_tables(sql, 10, 10, 10, 10);

    Sim_Param p;

    sim_param_write(sql, p);
    return 0;
}
