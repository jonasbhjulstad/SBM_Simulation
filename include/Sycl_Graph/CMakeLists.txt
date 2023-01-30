add_subdirectory(Graph)
add_library(Common STATIC path_config.cpp)
target_link_libraries(Common PRIVATE Eigen3::Eigen)
target_include_directories(Common INTERFACE "$<BUILD_INTERFACE:${SYCL_GRAPH_INCLUDE_DIRS}/..>")
