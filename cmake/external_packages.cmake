
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
# set(SYCL_INCLUDE_DIR /opt/intel/oneapi/compiler/2024.0/include/)
# set(SYCL_LIBRARY_DIR /opt/intel/oneapi/compiler/2024.0/lib/)
# include_directories(${SYCL_INCLUDE_DIR})
find_package(oneDPL REQUIRED HINTS "/opt/intel/oneapi/dpl/latest/lib/cmake/oneDPL/")
find_package(TBB REQUIRED HINTS "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb/")
find_package(Eigen3 REQUIRED)
include(FindOpenMP)
include(ExternalProject)
ExternalProject_Add(
        casadi-3.6.5
        URL https://github.com/casadi/casadi/releases/download/3.6.5/casadi-3.6.5-linux64-py39.zip
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        PREFIX ${CMAKE_BINARY_DIR}/external/casadi)

find_package(casadi HINTS ${CMAKE_BINARY_DIR}/external/casadi/src/casadi-3.6.5/casadi)
CPMAddPackage(NAME yaml-cpp
GITHUB_REPOSITORY jbeder/yaml-cpp
GIT_TAG master)
CPMFindPackage(NAME cppitertools
GITHUB_REPOSITORY ryanhaining/cppitertools
GIT_TAG master
OPTIONS
"cppitertools_INSTALL_CMAKE_DIR \"share/cppitertools/cmake\""
)
set(${PROJECT_NAME}_EXTERNAL_PRIVATE_LIBRARIES TBB::tbb cppitertools::cppitertools yaml-cpp::yaml-cpp)
set(${PROJECT_NAME}_EXTERNAL_TEST_LIBRARIES yaml-cpp::yaml-cpp)