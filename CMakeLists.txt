set(ENABLE_TESTING ON)
cmake_minimum_required(VERSION 3.5)
project(Sycl_Graph)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)
SET(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER_FORCED ON)
if(WIN32)
    set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
    set(PYTHON_EXECUTABLE "/home/man/.pyenv/versions/3.9.13/bin/python3")
    set(PYTHON_INCLUDE_DIR "/home/man/.pyenv/versions/3.9.13/include/python3.9")
    set(PYTHON_LIBS "/home/man/.pyenv/versions/3.9.13/libs/python3.9")
    set(Python3_ROOT_DIR "/home/man/.pyenv/versions/3.9.13/")
endif()
find_package(PkgConfig)
list(APPEND CMAKE_MODULE_PATH "/home/arch/Downloads/cmake-checks-cache/CMakeChecksCache/")
find_package(Eigen3 REQUIRED)
get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
set(HIPSYCL_TARGETS "cuda:sm_60")
set(SPIR_FORMAT __DEVICE_SPIR64__)
list(REMOVE_ITEM importTargetsAfter ${importTargets})
message("${importTargetsAfter}")
set(PYBIND11_PYTHON_VERSION "3.9")
find_package(pybind11 CONFIG)
# find_package(PythonLibs REQUIRED)
find_package(Python3 3.9 REQUIRED)
set(SYCL_GRAPH_USE_INTEL ON)
set(SYCL_GRAPH_USE_AMD OFF)
set(SYCL_GRAPH_USE_INTEL_SYCL ON)
set(SYCL_GRAPH_USE_HIPSYCL OFF)
message(WARNING ${COMPILER_SUPPORTS_SYCL})

include(FetchContent)
FetchContent_Declare(
    cppitertools_repo
    GIT_REPOSITORY https://github.com/ryanhaining/cppitertools.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(cppitertools_repo)

if(${SYCL_GRAPH_USE_INTEL_SYCL})
add_compile_definitions(SYCL_GRAPH_USE_INTEL_SYCL)
endif()
if (${SYCL_GRAPH_USE_INTEL_SYCL} OR ${SYCL_GRAPH_USE_HIPSYCL})
    set(SYCL_GRAPH_USE_SYCL ON)
    add_compile_definitions(SYCL_GRAPH_USE_SYCL)
endif()
if(${SYCL_GRAPH_USE_INTEL} OR ${SYCL_GRAPH_USE_AMD})
add_compile_definitions(SYCL_GRAPH_USE_INTEL)
find_package(TBB CONFIG REQUIRED)
find_package(oneDPL CONFIG REQUIRED)
set(ONEAPI_LIBRARIES TBB::tbb oneDPL)
endif()

include(cmake/HIP_Sycl_config.cmake)
include(cmake/Intel_Sycl_config.cmake)
include(cmake/Sycl_executable.cmake)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options("-fPIC")
# add_compile_options("-fsanitize=address")
# add_link_options("-fsanitize=address")
# add_compile_options("-lasan")
include(FetchContent)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
endif()


set(COMMON_LIBRARIES "Common")

set(SYCL_GRAPH_DATA_DIR "${PROJECT_SOURCE_DIR}/data" CACHE INTERNAL "SYCL_GRAPH_DATA_DIR")
set(SYCL_GRAPH_LOG_DIR "${PROJECT_SOURCE_DIR}/log" CACHE INTERNAL "SYCL_GRAPH_LOG_DIR")
set(SYCL_GRAPH_INCLUDE "${PROJECT_SOURCE_DIR}/include" CACHE INTERNAL "SYCL_GRAPH_INCLUDE")
set(SYCL_GRAPH_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/static" CACHE INTERNAL "SYCL_GRAPH_INCLUDE_DIRS")


add_subdirectory(include)
add_subdirectory(static)
set(SYCL_GRAPH_LIBRARIES ${LIBRARIES} DataFrame SYCL_GRAPH_Eigen)
configure_file("${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.hpp.in" "${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.hpp")
configure_file("${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.py.in" "${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.py")
add_subdirectory(test)

set(SYCL_GRAPH_COMMON_LIBS ${ONEAPI_LIBRARIES} ${HIPSYCL_LIBS})