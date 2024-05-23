
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
set(SYCL_INCLUDE_DIR /opt/intel/oneapi/compiler/2024.0/include/)
set(SYCL_LIBRARY_DIR /opt/intel/oneapi/compiler/2024.0/lib/)
include_directories(${SYCL_INCLUDE_DIR})
find_package(IntelSYCL REQUIRED HINTS "/opt/intel/oneapi/compiler/latest/lib/cmake/IntelSYCL/")
find_package(oneDPL REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
include(ExternalProject)
ExternalProject_Add(
        casadi-3.6.5
        URL https://github.com/casadi/casadi/releases/download/3.6.5/casadi-3.6.5-linux64-py39.zip
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        PREFIX ${CMAKE_BINARY_DIR}/external/casadi)

find_package(casadi HINTS ${CMAKE_BINARY_DIR}/external/casadi/src/casadi-3.6.5/casadi)

CPMFindPackage(NAME cppitertools
GITHUB_REPOSITORY ryanhaining/cppitertools
OPTIONS
"cppitertools_INSTALL_CMAKE_DIR \"share/cppitertools/cmake\""
)
set(SIR_SBM_EXTERNAL_LIBRARIES oneDPL TBB::tbb Eigen3::Eigen cppitertools::cppitertools)