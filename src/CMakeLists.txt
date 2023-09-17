

add_library(Graph STATIC Graph.cpp)
target_link_libraries(Graph PRIVATE cppitertools)
target_include_directories(Graph PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
add_custom_library(Regression)
target_link_libraries(Regression PRIVATE ortools::ortools nlohmann_json::nlohmann_json)
target_link_libraries(Regression PUBLIC Eigen3::Eigen)
target_include_directories(Regression PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
add_subdirectory(Utils)
add_subdirectory(Simulation)
add_custom_library(Dynamics)
target_link_libraries(Dynamics PRIVATE Sim_Types Sim_Buffers)
# target_link_libraries(Simulation PUBLIC Graph Sim_Dynamics Profiling)
