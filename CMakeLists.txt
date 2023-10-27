cmake_minimum_required(VERSION 3.22)

# ---- Project ----

# Note: update this to your new project's name and version
project(
  SBM_Graph
  VERSION 1.0
  LANGUAGES CXX
)
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

include(cmake/external_packages.cmake)
if(NOT ${IntelSYCL_FOUND})
find_package(IntelSYCL CONFIG REQUIRED)
endif()

set(SBM_SIMULATION_DEFAULT_FLAGS -Wno-deprecated-declarations -Wno-unknown-cuda-version)
include(cmake/default_targets.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SBM_GRAPH_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
# ---- Add source files ----
add_subdirectory(source)



string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

set(${PROJECT_NAME}_LIBRARY_TARGETS )
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
