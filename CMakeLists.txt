set(ENABLE_TESTING ON)
cmake_minimum_required(VERSION 3.5)
project(Sycl_Graph)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER_FORCED ON)
set(SYCL_HEADER_INCLUDE_DIR "/opt/intel/oneapi/compiler/latest/linux/include/sycl")
if(WIN32)
    set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
    set(PYTHON_EXECUTABLE "/home/man/.pyenv/versions/3.9.13/bin/python3")
    set(PYTHON_INCLUDE_DIR "/home/man/.pyenv/versions/3.9.13/include/python3.9")
    set(PYTHON_LIBS "/home/man/.pyenv/versions/3.9.13/libs/python3.9")
    set(Python3_ROOT_DIR "/home/man/.pyenv/versions/3.9.13/")
endif()
list(APPEND CMAKE_MODULE_PATH "/home/arch/Downloads/cmake-checks-cache/CMakeChecksCache/")
set(HIPSYCL_TARGETS "cuda:sm_60")
set(SPIR_FORMAT __DEVICE_SPIR64__)
set(PYBIND11_PYTHON_VERSION "3.9")
find_package(pybind11 CONFIG)
find_package(Python3 3.9 REQUIRED)
option(SYCL_GRAPH_USE_CUDA "Use CUDA" ON)
set(SYCL_CUDA_TARGET "")
option(SYCL_GRAPH_USE_INTEL "Use Intel" ON)
option(SYCL_GRAPH_USE_HIPSYCL "Use HIPSYCL" OFF)
option(SYCL_GRAPH_USE_ONEAPI "Use OneAPI" OFF)
option(ENABLE_PRECOMPILED_HEADERS "Enable Precompiled Headers" OFF)
option(DEFAULT_WARNING_SUPPRESSION "Default Warning Suppression" ON)
set(DEFAULT_WARNING_FLAGS -Wunknown-cuda-version)
option(TRACY_ENABLE "" ON)
option (TRACY_ON_DEMAND "" ON)
set(DPCPP_FLAGS -fsycl -Wno-linker-warnings)
set(SYCL_CUDA_FLAGS -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60)

include(cmake/external_packages.cmake)
include(cmake/custom_executable.cmake)
include(cmake/ccache.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#Sycl warning suppression flags

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options("-stdlib=libstdc++")
endif()


set(SYCL_GRAPH_DATA_DIR "${PROJECT_SOURCE_DIR}/data" CACHE INTERNAL "SYCL_GRAPH_DATA_DIR")
set(SYCL_GRAPH_LOG_DIR "${PROJECT_SOURCE_DIR}/log" CACHE INTERNAL "SYCL_GRAPH_LOG_DIR")
set(SYCL_GRAPH_INCLUDE "${PROJECT_SOURCE_DIR}/include" CACHE INTERNAL "SYCL_GRAPH_INCLUDE")
set(SYCL_GRAPH_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/static;${SYCL_HEADER_INCLUDE_DIR}" CACHE INTERNAL "SYCL_GRAPH_INCLUDE_DIRS")
file(MAKE_DIRECTORY ${SYCL_GRAPH_DATA_DIR}/SIR_sim)

include_directories(${SYCL_INCLUDE_DIR})
add_subdirectory(include)
add_subdirectory(static)
# set(SYCL_GRAPH_LIBRARIES DataFrame Eigen3::Eigen)
configure_file("${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.hpp.in" "${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.hpp")
configure_file("${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.py.in" "${PROJECT_SOURCE_DIR}/include/Sycl_Graph/path_config.py")
add_subdirectory(test)
add_subdirectory(Executables)

# Doxygen

# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs_doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target( docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)