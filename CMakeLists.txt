cmake_minimum_required(VERSION 3.22)

# ---- Project ----

# Note: update this to your new project's name and version
project(
  SBM_Simulation
  VERSION 1.0
  LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)
#export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()
# option(ENABLE_TBB "Enable TBB" ON)
option(ENABLE_CUDA "Enable CUDA" OFF)
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
# add_link_options("-v")
if (${ENABLE_CUDA})
    # add_link_options(--cuda-path="/home/man/cuda_remote")
    # add_compile_options(--cuda-path="/home/man/cuda_remote")
    set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda)
    # set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -fsycl-targets=nvidia_gpu_sm_86)
    # add_link_options(-fsycl -fsycl-targets=nvidia_gpu_sm_86)
    # add_compile_options(-Xsycl-target-backend=nvptx64-nvidia-cuda)
    # add_link_options(-Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86)
    # set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -Xcuda-ptxas "--verbose")
endif()
add_compile_options("-Wfatal-errors")
include(cmake/CPM.cmake)

set(PYTHON_ROOT "/home/man/mambaforge/envs/gt/")

if(WIN32)
    set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
    set(PYTHON_EXECUTABLE "${PYTHON_ROOT}/bin/python3")
    set(PYTHON_INCLUDE_DIR "${PYTHON_ROOT}/include/python3.9")
    set(PYTHON_LIBS "${PYTHON_ROOT}/libs/python3.9")
    set(Python3_ROOT_DIR "${PYTHON_ROOT}/")
endif()
set(PYBIND11_PYTHON_VERSION "3.9")
set(PYTHON_ENV_CMAKE_MODULE_DIR "/home/man/mambaforge/envs/gt/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
include(FindPython3)

find_package(Python3 COMPONENTS Interpreter Development)

find_package(pybind11 CONFIG HINTS ${PYTHON_ENV_CMAKE_MODULE_DIR} REQUIRED)
find_package(IntelSYCL CONFIG REQUIRED)
find_package(ortools CONFIG REQUIRED)
include(cmake/external_packages.cmake)


set(SBM_SIMULATION_DEFAULT_FLAGS -Wno-deprecated-declarations -Wno-unknown-cuda-version)
include(cmake/default_targets.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SBM_DATABASE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SBM_SIMULATION_DATA_DIR "${PROJECT_SOURCE_DIR}/data" CACHE INTERNAL "SBM_SIMULATION_DATA_DIR")
set(SBM_SIMULATION_LOG_DIR "${PROJECT_SOURCE_DIR}/log" CACHE INTERNAL "SBM_SIMULATION_LOG_DIR")
set(SBM_SIMULATION_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" CACHE INTERNAL "SBM_SIMULATION_INCLUDE_DIR")
configure_file("${PROJECT_SOURCE_DIR}/include/SBM_Simulation/Utils/path_config.hpp.in" "${PROJECT_SOURCE_DIR}/include/SBM_Simulation/Utils/path_config.hpp")

# ---- Add source files ----
add_subdirectory(source)
add_subdirectory(binders)
add_subdirectory(executables)



string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

set(${PROJECT_NAME}_LIBRARY_TARGETS ${PROJECT_NAME}_Types ${PROJECT_NAME}_Utils Epidemiological)
# install export for Utils
install(TARGETS ${${PROJECT_NAME}_LIBRARY_TARGETS}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES "Eigen3 3.4.0; Dataframe 1.0.0; soci 4.2.0"
)
