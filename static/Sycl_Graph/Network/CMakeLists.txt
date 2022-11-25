add_library(Network INTERFACE Graph_Generation.hpp SIR_Bernoulli_Network.hpp)
target_include_directories(Network INTERFACE ${SYCL_GRAPH_INCLUDE_DIRS})
target_link_libraries(Network INTERFACE Math cppitertools Sycl_Graph ${ONEAPI_LIBRARIES})
