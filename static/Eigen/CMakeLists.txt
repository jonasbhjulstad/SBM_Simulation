add_library(SYCL_GRAPH_Eigen STATIC SYCL_GRAPH_Eigen.cpp SYCL_GRAPH_Eigen.hpp)
target_link_libraries(SYCL_GRAPH_Eigen PUBLIC Common Eigen3::Eigen DataFrame)
target_include_directories(SYCL_GRAPH_Eigen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})