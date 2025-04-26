
include(${CMAKE_CURRENT_LIST_DIR}/configure_external.cmake)
find_package(TBB REQUIRED HINTS "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb/")
find_package(oneDPL REQUIRED HINTS "/opt/intel/oneapi/dpl/latest/lib/cmake/oneDPL/")
find_package(Eigen3 REQUIRED)
include(FindOpenMP)
ExternalProject_MakeAvailable(
        NAME casadi
        LIB_NAME casadi
        URL https://github.com/casadi/casadi/releases/download/3.6.5/casadi-3.6.5-linux64-py39.zip)

ExternalProject_MakeAvailable(
        NAME yaml-cpp
        LIB_NAME yaml-cpp::yaml-cpp
        GIT_REPO https://github.com/jbeder/yaml-cpp.git
        TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/dpcpp.cmake
        CMAKE_EXTRA_ARGS
        -DYAML_CPP_BUILD_TESTS=OFF
        BUILD_TYPE Release
)

ExternalProject_MakeAvailable(
        NAME DataFrame
        LIB_NAME DataFrame::DataFrame
        GIT_REPO https://github.com/hosseinmoein/DataFrame.git
        TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/dpcpp.cmake
        BUILD_TYPE Release
)
ExternalProject_MakeAvailable(
        NAME cppitertools
        LIB_NAME cppitertools::cppitertools
        GIT_REPO https://github.com/ryanhaining/cppitertools.git
        TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/dpcpp.cmake
        BUILD_TYPE Release
)

set(${PROJECT_NAME}_EXTERNAL_PRIVATE_LIBRARIES TBB::tbb cppitertools::cppitertools yaml-cpp::yaml-cpp DataFrame::DataFrame)
set(${PROJECT_NAME}_EXTERNAL_TEST_LIBRARIES yaml-cpp::yaml-cpp DataFrame::DataFrame)

if(${${PROJECT_NAME}_ENABLE_GRAPH_TOOL})
include(FindPkgConfig)
pkg_check_modules(GRAPH_TOOL REQUIRED graph-tool-py3.12)
endif()