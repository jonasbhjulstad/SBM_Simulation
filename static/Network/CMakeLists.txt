add_library(Network INTERFACE Graph_Generation.hpp SIR_Bernoulli_Network.hpp)
target_include_directories(Network INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Network INTERFACE Sycl_Graph_Math cppitertools FROLS_Graph ${ONEAPI_LIBRARIES})
