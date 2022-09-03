set(ENABLE_TESTING OFF)
cmake_minimum_required(VERSION 3.5)
project(FROLS)
find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)
find_package(SUNDIALS REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(SUNDIALS_LIBS SUNDIALS::cvode SUNDIALS::nvecserial)
set(DEFAULT_LIBS Eigen3::Eigen ${SUNDIALS_LIBS} igraph::igraph)
# set(CMAKE_CXX_FLAGS "-Wfatal-errors")
include(FetchContent)
if (ENABLE_TESTING)
FetchContent_Declare(
googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)
endif()

# find_package(sciplot REQUIRED)


find_package(igraph REQUIRED)
  
set(FROLS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FROLS_INCLUDE_DIRS ${FROLS_ROOT_DIR}/include ${FROLS_ROOT_DIR}/static)
set(FROLS_DATA_DIR ${FROLS_ROOT_DIR}/data)
configure_file(include/FROLS_Path_Config.hpp.in include/FROLS_Path_Config.hpp)
# For Windows: Prevent overriding the parent project's compiler/linker settings
add_executable(ex ex.cpp)
target_link_libraries(ex PRIVATE Eigen3::Eigen casadi)
add_subdirectory(static)
if (ENABLE_TESTING)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(test)
endif()
add_subdirectory(Executables)