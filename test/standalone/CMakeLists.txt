cmake_minimum_required(VERSION 3.17)
project(kernel_test)

add_executable(kernel_test kernel_test.cpp)
target_compile_options(kernel_test PUBLIC ${${PROJECT_NAME}_SYCL_FLAGS})
target_link_options(kernel_test PUBLIC ${${PROJECT_NAME}_SYCL_FLAGS} -Wno-unknown-cuda-version)