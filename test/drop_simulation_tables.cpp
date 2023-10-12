#include <Sycl_Graph/Database/Simulation_Tables.hpp>



int main()
{
    pqxx::connection con("dbname=postgres user=postgres password=postgres");
    drop_simulation_tables(con);
    return 0;
}
