#default cmakelists
cmake_minimum_required(VERSION 3.22)

#c++20
set(CMAKE_CXX_STANDARD 20)
project(SIR_SBM)
set(SIR_SBM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)
set(SIR_SBM_LIBRARIES "")
add_compile_options("-fsycl")
add_compile_options("-fcolor-diagnostics")
option(SIR_SBM_USE_GPU "Use GPU" OFF)
#set policy CMP0125 to new
cmake_policy(SET CMP0135 NEW)
include(cmake/external_packages.cmake)
include(cmake/default_targets.cmake)
#export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#make dir
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/SIR_SBM/)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/SIR_SBM/)
set(SIR_SBM_DEV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/)
set(SIR_SBM_INCLUDE_DIR ${PROJECT_BINARY_DIR}/include/)
set(CMAKE_CXX_STANDARD 20)
set(SIR_SBM_LIBRARY_INCLUDE_DIRS ${SIR_SBM_INCLUDE_DIR} ${SYCL_INCLUDE_DIR} /usr/include/c++/11/)
file(GLOB DEV_HEADERS ${SIR_SBM_DEV_INCLUDE_DIR}/*.hpp)
#get ${PROJECT_BINARY_DIR}/include/SIR_SBM/dev_header_name.hpp/.cpp names
include(${PROJECT_SOURCE_DIR}/cmake/lzz_routines.cmake)


#LZZ generation
add_lzz_library(SIR_SBM "${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/")
add_lzz_library(SIR_SBM_Regression "${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/regression/")
target_link_libraries(SIR_SBM_Regression PRIVATE SIR_SBM casadi)
target_include_directories(SIR_SBM PUBLIC ${SIR_SBM_INCLUDE_DIR} ${SIR_SBM_EXTERNAL_INCLUDE_DIRS})
target_link_options(SIR_SBM PUBLIC "-fsycl")

add_subdirectory(test)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)