add_library(Math STATIC math.cpp)
target_link_libraries(Math PUBLIC Graph Common)
target_include_directories(Math PUBLIC "$<BUILD_INTERFACE:${SYCL_GRAPH_INCLUDE_DIRS}>")
