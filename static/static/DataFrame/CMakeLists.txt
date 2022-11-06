add_library(DataFrame STATIC DataFrame.cpp DataFrame.hpp ../Graph/FROLS_Graph.hpp)
target_include_directories(DataFrame PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(DataFrame PUBLIC Sycl_Graph)