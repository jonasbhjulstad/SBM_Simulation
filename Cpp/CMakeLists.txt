set(ENABLE_TESTING ON)
cmake_minimum_required(VERSION 3.5)
project(FROLS)
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)
SET(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER_FORCED ON)
if(WIN32)
    set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
    # set(PYTHON_EXECUTABLE "/home/man/Env/Default/bin/python")
endif()
find_package(PkgConfig)
list(APPEND CMAKE_MODULE_PATH "/home/arch/Downloads/cmake-checks-cache/CMakeChecksCache/")
find_package(Eigen3 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(ZLIB REQUIRED)
find_package(fmt REQUIRED)
find_package(re2 REQUIRED)
find_package(ortools CONFIG HINTS "")
get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
find_package(range-v3 REQUIRED)
set(HIPSYCL_TARGETS "cuda:sm_60")
set(SPIR_FORMAT __DEVICE_SPIR64__)
list(REMOVE_ITEM importTargetsAfter ${importTargets})
message("${importTargetsAfter}")
find_package(SUNDIALS REQUIRED)
find_package(pybind11 CONFIG)
set(FROLS_USE_INTEL OFF)
set(FROLS_USE_INTEL_SYCL OFF)
set(FROLS_USE_HIPSYCL OFF)

if(${FROLS_USE_INTEL_SYCL})
add_compile_definitions(FROLS_USE_INTEL_SYCL)
endif()
if (${FROLS_USE_INTEL_SYCL} OR ${FROLS_USE_HIPSYCL})
    set(FROLS_USE_SYCL ON)
    add_compile_definitions(FROLS_USE_SYCL)
endif()
if(${FROLS_USE_INTEL})
add_compile_definitions(FROLS_USE_INTEL)
find_package(TBB REQUIRED)
set(ONEAPI_LIBS TBB::tbb)
endif()

include(cmake/HIP_Sycl_config.cmake)
include(cmake/Intel_Sycl_config.cmake)
include(cmake/Sycl_executable.cmake)


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

set(COMMON_LIBRARIES "Common;FROLS_Math;DataFrame;FROLS_Eigen;cppitertools")

# set(GRAPH_LITE_SUPPORT_DOT OFF)
# FetchContent_Declare(graphlite_repo
#     GIT_REPOSITORY https://github.com/haasdo95/graphlite.git
#     GIT_TAG master)

# FetchContent_MakeAvailable(graphlite_repo)
# set_target_properties(graphlite
#     PROPERTIES
#     CXX_STANDARD 17)

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

set(FROLS_COMMON_LIBS DataFrame Common FROLS_Math FROLS_Eigen FROLS_Network ${ONEAPI_LIBS} ${HIPSYCL_LIBS})

add_subdirectory(Executables)