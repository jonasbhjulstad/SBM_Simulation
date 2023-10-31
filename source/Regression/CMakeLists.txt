add_default_library(Regression)
target_link_libraries(Regression PUBLIC ortools::ortools SBM_Graph::SBM_Graph)
target_link_libraries(Regression PUBLIC Eigen3::Eigen math SBM_Database)
target_include_directories(Regression PUBLIC ${SBM_SIMULATION_INCLUDE_DIR})
