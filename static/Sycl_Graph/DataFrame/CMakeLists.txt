add_library(DataFrame STATIC DataFrame.cpp DataFrame.hpp)
target_include_directories(DataFrame PUBLIC ${SYCL_GRAPH_INCLUDE_DIRS})
target_link_libraries(DataFrame PUBLIC Common)