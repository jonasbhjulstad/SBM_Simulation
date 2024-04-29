
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
set(SYCL_INCLUDE_DIR /opt/intel/oneapi/compiler/2024.0/include/)
set(SYCL_LIBRARY_DIR /opt/intel/oneapi/compiler/2024.0/lib/)
include_directories(${SYCL_INCLUDE_DIR})
find_package(IntelSYCL REQUIRED HINTS "/opt/intel/oneapi/compiler/latest/lib/cmake/IntelSYCL/")
find_package(oneDPL REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)


CPMFindPackage(NAME cppitertools
GITHUB_REPOSITORY ryanhaining/cppitertools
OPTIONS
"cppitertools_INSTALL_CMAKE_DIR \"share/cppitertools/cmake\""
)
set(SIR_SBM_EXTERNAL_LIBRARIES oneDPL TBB::tbb Eigen3::Eigen cppitertools::cppitertools)