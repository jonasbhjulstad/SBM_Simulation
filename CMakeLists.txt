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


include(cmake/external_packages.cmake)
include(cmake/default_targets.cmake)

#make dir
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/SIR_SBM/)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/SIR_SBM/)
set(SIR_SBM_DEV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/dev_headers/SIR_SBM/)
set(SIR_SBM_INCLUDE_DIR ${PROJECT_BINARY_DIR}/include/)

set(SIR_SBM_LIBRARY_INCLUDE_DIRS ${SIR_SBM_INCLUDE_DIR} ${SYCL_INCLUDE_DIR} /usr/include/c++/11/)
file(GLOB DEV_HEADERS ${SIR_SBM_DEV_INCLUDE_DIR}/*.hpp)
#get ${PROJECT_BINARY_DIR}/include/SIR_SBM/dev_header_name.hpp/.cpp names
set(SIR_SBM_SOURCE_FILES "")
set(SIR_SBM_HEADER_FILES "")
foreach(DEV_HEADER ${DEV_HEADERS})
    #get basename no ext
    get_filename_component(DEV_HEADER_NAME ${DEV_HEADER} NAME_WE)
    set(SOURCE_NAME ${CMAKE_BINARY_DIR}/include/SIR_SBM/${DEV_HEADER_NAME}.cpp)
    set(HEADER_NAME ${CMAKE_BINARY_DIR}/include/SIR_SBM/${DEV_HEADER_NAME}.hpp)
    list(APPEND SIR_SBM_SOURCE_FILES ${SOURCE_NAME})
    list(APPEND SIR_SBM_HEADER_FILES ${HEADER_NAME})
endforeach()
message(WARNING ${SYCL_INCLUDE_DIR})
add_custom_command(
    OUTPUT ${SIR_SBM_SOURCE_FILES} ${SIR_SBM_HEADER_FILES}
    #-I/usr/include/c++/11/ 
    COMMAND lzz -sx cpp -hx hpp -hd -sd ${DEV_HEADERS} 
    -I${PROJECT_SOURCE_DIR}/dev_headers/ 
    # -I${SYCL_INCLUDE_DIR} 
    # -I${SYCL_INCLUDE_DIR}/sycl/
    # -I/opt/intel/oneapi/tbb/2021.11/env/../include
    # -I/opt/intel/oneapi/dpl/2022.3/include
    # -I/opt/intel/oneapi/dev-utilities/2024.0/include
    # -I/opt/intel/oneapi/compiler/2024.0/opt/oclfpga/include
    # -I/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11
    # -I/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/x86_64-linux-gnu/c++/11
    # -I/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/backward
    # -I/opt/intel/oneapi/compiler/2024.0/lib/clang/17/include
    # -I/usr/local/include
    # -I/usr/include/x86_64-linux-gnu
    # -I/usr/include
    -o ${CMAKE_BINARY_DIR}/include/SIR_SBM/
    DEPENDS ${DEV_HEADERS}
)

file(GLOB SIR_SBM_SOURCE_FILES ${PROJECT_BINARY_DIR}/include/SIR_SBM/*.*pp)

add_library(SIR_SBM ${SIR_SBM_SOURCE_FILES})
target_include_directories(SIR_SBM PUBLIC ${SIR_SBM_INCLUDE_DIR})
target_link_options(SIR_SBM PUBLIC "-fsycl")

add_subdirectory(test)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)