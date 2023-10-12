add_custom_library(Regression)
target_link_libraries(Regression PUBLIC ortools::ortools nlohmann_json::nlohmann_json Graph)
target_link_libraries(Regression PUBLIC Eigen3::Eigen math)
target_include_directories(Regression PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
