add_subdirectory(Graph)
add_library(Common STATIC path_config.cpp)
target_include_directories(Common INTERFACE "$<BUILD_INTERFACE:${SYCL_GRAPH_INCLUDE_DIRS}/..>")
target_link_libraries(Common INTERFACE Eigen3::Eigen)
