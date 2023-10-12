#include <Sycl_Graph/Database/Simulation_Tables.hpp>



int main()
{
    pqxx::connection con("dbname=postgres user=postgres password=postgres");
    drop_simulation_tables(con);
    construct_simulation_tables(con, 10, 10, 10, 10);

    Sim_Param p;

    sim_param_write(con, p);
    return 0;
}
