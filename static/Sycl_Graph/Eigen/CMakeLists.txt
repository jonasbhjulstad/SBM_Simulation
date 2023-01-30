add_library(Sycl_Graph_Eigen STATIC Sycl_Graph_Eigen.cpp Sycl_Graph_Eigen.hpp)
target_link_libraries(Sycl_Graph_Eigen PUBLIC Graph Eigen3::Eigen DataFrame)
target_include_directories(Sycl_Graph_Eigen PUBLIC ${SYCL_GRAPH_INCLUDE_DIRS})