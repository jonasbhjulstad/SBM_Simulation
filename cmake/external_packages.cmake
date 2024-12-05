
include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)
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

function(Custom_ExternalProject_Add name git_url)
    ExternalProject_Add(
            ${name}_repo
            GIT_REPOSITORY ${git_url}
            PREFIX ${CMAKE_BINARY_DIR}/external/${name}
            LOG_INSTALL "false"
            INSTALL_COMMAND ""
            CMAKE_ARGS "-DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/cmake/toolchains/dpcpp.cmake"
    )
    set(${name}_BINARY_DIR ${CMAKE_BINARY_DIR}/external/${name}_repo/src/${name})
    find_package(${name} HINTS ${${name}_BINARY_DIR})
endfunction()

Custom_ExternalProject_Add(
        yaml-cpp
        https://github.com/jbeder/yaml-cpp.git
)

Custom_ExternalProject_Add(
        DataFrame
        https://github.com/jbeder/yaml-cpp.git
)
Custom_ExternalProject_Add(
        cppitertools
        https://github.com/ryanhaining/cppitertools.git
)

set(${PROJECT_NAME}_EXTERNAL_PRIVATE_LIBRARIES TBB::tbb cppitertools::cppitertools yaml-cpp::yaml-cpp DataFrame::DataFrame)
set(${PROJECT_NAME}_EXTERNAL_TEST_LIBRARIES yaml-cpp::yaml-cpp DataFrame::DataFrame)

if(${${PROJECT_NAME}_ENABLE_GRAPH_TOOL})
include(FindPkgConfig)
pkg_check_modules(GRAPH_TOOL REQUIRED graph-tool-py3.12)
endif()