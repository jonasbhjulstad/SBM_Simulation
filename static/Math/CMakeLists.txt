add_library(Math STATIC Sycl_Graph_Math.cpp)
target_link_libraries(Math PUBLIC Common DataFrame SYCL_Graph_Sycl)
target_include_directories(Math PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
