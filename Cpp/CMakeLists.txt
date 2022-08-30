cmake_minimum_required(VERSION 3.5)
project(Eigen_FROLS)
find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)
find_package(SUNDIALS REQUIRED)

find_package(OpenMP)
set(CMAKE_CXX_STANDARD 17)
set(SUNDIALS_LIBS SUNDIALS::cvode SUNDIALS::nvecserial)
set(DEFAULT_LIBS Eigen3::Eigen ${SUNDIALS_LIBS} igraph::igraph)

include(FetchContent)
FetchContent_Declare(
googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG release-1.12.1
)

find_package(DataFrame QUIET)
if (NOT DataFrame_FOUND)
FetchContent_Declare(
dataframe_repo
GIT_REPOSITORY https://github.com/hosseinmoein/DataFrame.git
)
FetchContent_MakeAvailable(dataframe_repo)
endif()
  
find_package(igraph REQUIRED)
  
set(FROLS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FROLS_INCLUDE_DIR ${FROLS_ROOT_DIR}/include)
set(FROLS_DATA_DIR ${FROLS_ROOT_DIR}/data)
configure_file(${FROLS_INCLUDE_DIR}/FROLS_Path_Config.hpp.in ${FROLS_INCLUDE_DIR}/FROLS_Path_Config.hpp)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_executable(ex ex.cpp)
target_link_libraries(ex PRIVATE Eigen3::Eigen casadi)

add_subdirectory(test)
add_subdirectory(Executables)