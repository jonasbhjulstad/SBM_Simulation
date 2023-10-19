#include <doctest/doctest.h>
#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <filesystem>
#include <fstream>
#include <array>
#include <vector>
#include <string>
int simulation_table_test()
{
    //get cwd
    std::filesystem::path cwd = std::filesystem::current_path();
    using namespace SBM_Database;
    soci::session sql("postgresql", "user=postgres password=postgres");
    std::ofstream log_file(std::string(cwd) + "/log.txt");
    sql.set_log_stream(&log_file);
    std::vector<std::string> indices({"N0", "N1", "N2"});
    std::array<std::string, 3> data_names({"int0", "int1", "float"});
    std::array<std::string, 3> data_types({"integer", "integer", "real"});
// void construct_simulation_tables(soci::session &sql, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt);
    uint32_t Np = 2;
    uint32_t Ng = 2;
    uint32_t Ns = 2;
    uint32_t Nt = 10;

    construct_simulation_tables(sql, Np, Ng, Ns, Nt);

    drop_simulation_tables(sql);

    return 0;

}
TEST_CASE("Simulation_Tables")
{
    simulation_table_test();
}
