#include <Sycl_Graph/Database/Dataframe.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Types.hpp>
#include <Sycl_Graph/Database/Table.hpp>


int main()
{
    pqxx::connection con("dbname=postgres user=postgres password=postgres");

    std::vector<std::string> indices({"N0", "N1", "N2"});
    std::vector<std::string> data_names({"int0", "int1", "float"});
    std::vector<std::string> data_types({"integer", "integer", "real"});

    create_table(con, "test_table", indices, data_names, data_types);

    table_insert(con, "test_table", {"N0", "N1", "N2"}, {0,0,0}, {"int0", "int1", "float"}, std::make_tuple(1,2,3.0f));

    auto ints = read_table_column<int, float>(con, "test_table", {"N0", "N1", "N2"}, {0,0,0}, {"int0", "float"}, {"integer", "real"});

    table_insert(con, "test_table", {"N0", "N1", "N2"}, {1,0,0}, {"int0", "int1", "float"}, std::make_tuple(12,100,4.3f));
    table_insert(con, "test_table", {"N0", "N1", "N2"}, {2,0,0}, {"int0", "int1", "float"}, std::make_tuple(112,1020,42.3f));

    auto ints_vec = read_table_slice<int, int, float>(con, "test_table", {"N1", "N2"}, {0,0}, "N0", {0,2}, {"int0", "int1", "float"}, {"integer", "integer", "real"});

    drop_table(con, "test_table");
    return 0;

}
