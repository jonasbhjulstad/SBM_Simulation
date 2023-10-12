#ifndef SYCL_GRAPH_DATABASE_REGRESSION_TABLES_HPP
#define SYCL_GRAPH_DATABASE_REGRESSION_TABLES_HPP
#include <Sycl_Graph/Database/Enum.hpp>
#include <Sycl_Graph/Database/Table.hpp>

void create_regression_table(pqxx::connection& con)
{

    define_enum(con, "Community_t", {"Detected_Communities", "True_Communities"});
    define_enum(con, "Regression_t", {"LS", "QR"});
    std::vector<std::string> theta_indices({"p_out", "graph"});
    std::vector<std::string> theta_data_names({"theta", "fitness", "Regression-Type", "Community-Type"});
    std::vector<std::string> theta_data_types({"real[]", "real[]", "Regression_t", "Community_t"});
    create_table(con, "regression_parameters", theta_indices, theta_data_names, theta_data_types);
}

void regression_table_write(pqxx::work& work, uint32_t p_out, uint32_t graph, const std::vector<float>& theta, const std::vector<float>& fitness, Community_Type c_type, Regression_Type r_type)
{
  std::string insert_str = "INSERT INTO connection_community_map (p_out, graph, theta, fitness, Regression-Type, Community-Type) VALUES (" + std::to_string(p_out) + ", " + std::to_string(graph) + ", ";
  std::string command = insert_str + vector_to_string(theta) + ", " + vector_to_string(fitness) + ", " + to_string(r_type) + ", " + to_string(c_type) + ");";
  work.exec(command);
}

void drop_regression_table(pqxx::connection& con)
{
    drop_table(con, "regression_parameters");
}


#endif
