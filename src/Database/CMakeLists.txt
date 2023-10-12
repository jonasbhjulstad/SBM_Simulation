add_custom_library(Database)
target_include_directories(Database PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
#pqxx
target_link_libraries(Database PUBLIC libpqxx::pqxx Eigen3::Eigen math Dataframe String_Manipulation Sim_Types)
