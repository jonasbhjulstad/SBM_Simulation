#include <Sycl_Graph/Database/Dataframe.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <Sycl_Graph/Database/Table.hpp>


int main()
{
    pqxx::connection con("dbname=postgres user=postgres password=postgres");

    std::vector<std::string> indices({"N0", "N1", "N2"});
    std::vector<std::string> data_names({"int0", "int1", "float"});
    std::vector<std::string> data_types({"integer", "integer", "real"});

    create_table(con, "test_table", indices, data_names, data_types);

    // pqxx::connection &con, const std::string &table_name,
                            //  const std::vector<std::string> &index_names,
                            //  const std::vector<uint32_t> &index_values,
                            //  const std::vector<std::string> &data_names)
// template <typename... Ts>
// void table_insert(pqxx::work &work, const std::string &table_name,
//                   const std::vector<std::string> &index_names,
//                   const std::vector<uint32_t> &index_values,
//                   const std::vector<std::string> &data_names,
//                   const std::tuple<Ts...> &data)

    table_insert(con, "test_table", {"N0", "N1", "N2"}, {0,0,0}, {"int0", "int1", "float"}, std::make_tuple(1,2,3.0f));

    auto ints = read_table<int, int>(con, "test_table", {"N0", "N1", "N2"}, {0,0,0}, {"int0", "int1"});

    drop_table(con, "test_table");
    return 0;

}
