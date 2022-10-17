set(ENABLE_TESTING ON)
cmake_minimum_required(VERSION 3.5)
project(FROLS)

if(WIN32)
    set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
    set(PYTHON_EXECUTABLE "/home/man/Env/Default/bin/python")
endif()

list(APPEND CMAKE_MODULE_PATH "/home/arch/Downloads/cmake-checks-cache/CMakeChecksCache/")
find_package(Eigen3 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(ZLIB REQUIRED)
find_package(fmt REQUIRED)

find_package(ortools CONFIG HINTS "")
get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
find_package(range-v3 REQUIRED)

list(REMOVE_ITEM importTargetsAfter ${importTargets})
message("${importTargetsAfter}")
find_package(SUNDIALS REQUIRED)
find_package(pybind11 CONFIG)
set(FROLS_USE_INTEL_SYCL OFF)
set(FROLS_USE_HIPSYCL OFF)
find_package(TBB)
set(ONEAPI_LIBS TBB::tbb)

if(${FROLS_USE_INTEL_SYCL})
    # get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
    # get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
    find_package(IntelDPCPP REQUIRED HINTS "/opt/intel/oneapi/compiler/latest/linux/IntelDPCPP")
    set(TBB_DIR "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb
/")
    find_package(TBB REQUIRED)
    
    find_package(oneDPL REQUIRED HINTS "/opt/intel/oneapi/dpl/2021.7.0/lib/cmake/oneDPL/")

    set(ONEAPI_LIBS oneDPL TBB::tbb)

    # list(REMOVE_ITEM importTargetsAfter ${importTargets})
    # message(WARNING "${importTargetsAfter}")
    # set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/2022.1.0/linux/bin/icpx)
    # set(CMAKE_C_COMPILER /opt/intel/oneapi/compiler/2022.1.0/linux/bin/icx)
    add_compile_definitions(FROLS_USE_INTEL_SYCL)
    add_compile_definitions(__DEVICE_SPIR64__)
    add_compile_options("-fsycl")
    add_compile_definitions(FROLS_USE_SYCL)
    set(FROLS_USE_SYCL ON)

elseif(${FROLS_USE_HIPSYCL})
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)
    find_package(HIPSYCL REQUIRED)

    if(NOT HIPSYCL_DEBUG_LEVEL)
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(HIPSYCL_DEBUG_LEVEL 3 CACHE INTEGER
                "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                FORCE)
        else()
            set(HIPSYCL_DEBUG_LEVEL 2 CACHE INTEGER
                "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                FORCE)
        endif()
    endif()

    # add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
    # Use add_definitions for now for older cmake versions
    cmake_policy(SET CMP0005 NEW)
    add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

    if(WIN32)
        add_definitions(-D_USE_MATH_DEFINES)
    endif()

    add_compile_definitions(FROLS_USE_HIPSYCL)
    add_compile_definitions(FROLS_USE_SYCL)
    set(FROLS_USE_SYCL ON)

    # add_compile_definitions(__HIPSYCL_ENABLE_SYCL_TARGET__)
    add_compile_options("-fsycl")

    endif()

# find_package(nlohmann_json REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(UNIX)
# cmake_policy(SET CMP0069 NEW)
# set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
# endif()
set(SUNDIALS_LIBS SUNDIALS::cvode SUNDIALS::nvecserial)
include(FetchContent)

if(ENABLE_TESTING)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# option(cppitertools_INSTALL_CMAKE_DIR "share/cppitertools/cmake")
# find_package(cppitertools QUIET)
find_package(GTest REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

    # set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*; --export-fixes=out.yml")
endif()

FetchContent_Declare(
    cppitertools_repo
    GIT_REPOSITORY https://github.com/ryanhaining/cppitertools.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(cppitertools_repo)

set(COMMON_LIBRARIES "Common;FROLS_Math;DataFrame;FROLS_Eigen;cppitertools;graphlite")

set(GRAPH_LITE_SUPPORT_DOT OFF)
FetchContent_Declare(graphlite_repo
    GIT_REPOSITORY https://github.com/haasdo95/graphlite.git
    GIT_TAG master)

FetchContent_MakeAvailable(graphlite_repo)
set_target_properties(graphlite
    PROPERTIES
    CXX_STANDARD 17)

set(FROLS_DATA_DIR "${PROJECT_SOURCE_DIR}/data" CACHE INTERNAL "FROLS_DATA_DIR")
set(FROLS_LOG_DIR "${PROJECT_SOURCE_DIR}/log" CACHE INTERNAL "FROLS_LOG_DIR")
set(FROLS_INCLUDE "${PROJECT_SOURCE_DIR}/include" CACHE INTERNAL "FROLS_INCLUDE")
set(FROLS_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/static;${PROJECT_SOURCE_DIR}/static/Features;${PROJECT_SOURCE_DIR}/static/Algorithm;${PROJECT_SOURCE_DIR}/static/Regressors" CACHE INTERNAL "FROLS_INCLUDE_DIRS")

add_subdirectory(include)
add_subdirectory(static)
set(FROLS_LIBRARIES ${LIBRARIES} DataFrame FROLS_Eigen Regressors Features)
add_subdirectory(Binders)
configure_file("${PROJECT_SOURCE_DIR}/include/FROLS_Path_Config.cpp.in" "${PROJECT_SOURCE_DIR}/include/FROLS_Path_Config.cpp")

# For Windows: Prevent overriding the parent project's compiler/linker settings
if(ENABLE_TESTING)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(test)
    message(STATUS "Testing enabled")
endif()

add_subdirectory(Executables)