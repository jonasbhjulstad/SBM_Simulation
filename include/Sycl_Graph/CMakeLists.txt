add_library(Sycl_Graph STATIC Math/math.cpp)
target_include_directories(Sycl_Graph PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(Sycl_Graph PRIVATE Tracy::TracyClient)
target_link_libraries(Sycl_Graph PUBLIC cppitertools::cppitertools)