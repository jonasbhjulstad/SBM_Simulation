add_subdirectory(Graph)
add_library(Sycl_Graph STATIC path_config.cpp)
target_link_libraries(Sycl_Graph PRIVATE Eigen3::Eigen)
target_include_directories(Sycl_Graph INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>")