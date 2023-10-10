add_library(Database Dataframe.cpp Timeseries.cpp Simulation_Tables.cpp Table.cpp)
target_include_directories(Database PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
#pqxx
target_link_libraries(Database PUBLIC pqxx Eigen3::Eigen math Dataframe)
