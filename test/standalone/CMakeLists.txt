cmake_minimum_required(VERSION 3.17)
project(SBM_Simulation_Standalone_Tests)
find_package(IntelSYCL REQUIRED)
include(cmake/Sycl_Config.cmake)

add_executable(kernel_test kernel_test.cpp)
target_compile_options(kernel_test PUBLIC ${${PROJECT_NAME}_SYCL_FLAGS})
target_link_options(kernel_test PUBLIC ${${PROJECT_NAME}_SYCL_FLAGS} -Wno-unknown-cuda-version)
target_include_directories(kernel_test PUBLIC $<BUILD_INTERFACE:${SYCL_INCLUDE_DIR}>)

add_executable(async_test async_test.cpp)