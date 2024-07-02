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
include_directories(${PROJECT_BINARY_DIR}/include)

function(add_lzz_sublibrary subfolder_name)
add_lzz_library(SIR_SBM_${subfolder_name} "${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/${subfolder_name}")
target_link_libraries(SIR_SBM_${subfolder_name} PUBLIC ${SIR_SBM_EXTERNAL_LIBRARIES})
target_include_directories(SIR_SBM_${subfolder_name} PUBLIC ${SIR_SBM_INCLUDE_DIR} ${SIR_SBM_EXTERNAL_INCLUDE_DIRS})
endfunction()

add_lzz_sublibrary(utils)
add_lzz_sublibrary(vector)
add_lzz_sublibrary(regression)
target_link_libraries(SIR_SBM_regression PUBLIC casadi)
add_lzz_sublibrary(simulation)
add_lzz_sublibrary(epidemiological)
target_link_libraries(SIR_SBM_epidemiological PUBLIC SIR_SBM_utils)
target_link_libraries(SIR_SBM_simulation PUBLIC SIR_SBM_vector SIR_SBM_epidemiological)
add_lzz_sublibrary(sycl)
target_compile_options(SIR_SBM_sycl PUBLIC "-fsycl")
add_lzz_sublibrary(graph)
target_link_libraries(SIR_SBM_graph PUBLIC SIR_SBM_utils)


add_lzz_library(SIR_SBM "${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/")
target_link_libraries(SIR_SBM PUBLIC ${SIR_SBM_EXTERNAL_LIBRARIES})
target_include_directories(SIR_SBM PUBLIC ${SIR_SBM_INCLUDE_DIR} ${SIR_SBM_EXTERNAL_INCLUDE_DIRS})



add_lzz_library(SIR_SBM_Regression "${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/regression/")
target_link_libraries(SIR_SBM_Regression PRIVATE SIR_SBM casadi)
# target_link_options(SIR_SBM PUBLIC "-fsycl")

add_subdirectory(test)
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)