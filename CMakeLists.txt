#default cmakelists
cmake_minimum_required(VERSION 3.22)

#c++20
project(SIR_SBM)
add_compile_options("-fsycl")
add_compile_options("-fcolor-diagnostics")
add_compile_options("-Wfatal-errors")
option(SIR_SBM_USE_GPU "Use GPU" OFF)
cmake_policy(SET CMP0135 NEW)
include(cmake/external_packages.cmake)
include(cmake/default_targets.cmake)

#Core Library
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_EXTERNAL_LIBRARIES})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
add_subdirectory(test)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)