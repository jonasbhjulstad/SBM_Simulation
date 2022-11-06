add_library(Math STATIC Graph_Math.cpp)
target_link_libraries(Math PUBLIC Sycl_Graph Common DataFrame)
target_include_directories(Math PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
