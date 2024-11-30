
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
find_package(TBB REQUIRED HINTS "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb/")
find_package(oneDPL REQUIRED HINTS "/opt/intel/oneapi/dpl/latest/lib/cmake/oneDPL/")
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
CPMFindPackage(NAME yaml-cpp
GITHUB_REPOSITORY jbeder/yaml-cpp
GIT_TAG master)
CPMFindPackage(NAME cppitertools
GITHUB_REPOSITORY ryanhaining/cppitertools
GIT_TAG master
OPTIONS
"cppitertools_INSTALL_CMAKE_DIR \"share/cppitertools/cmake\""
)
CPMFindPackage(NAME DataFrame
GITHUB_REPOSITORY hosseinmoein/DataFrame
GIT_TAG 3.3.0
)

set(${PROJECT_NAME}_EXTERNAL_PRIVATE_LIBRARIES TBB::tbb cppitertools::cppitertools yaml-cpp::yaml-cpp DataFrame::DataFrame)
set(${PROJECT_NAME}_EXTERNAL_TEST_LIBRARIES yaml-cpp::yaml-cpp)

if(${${PROJECT_NAME}_ENABLE_GRAPH_TOOL})
include(FindPkgConfig)
pkg_check_modules(GRAPH_TOOL REQUIRED graph-tool-py3.12)
endif()