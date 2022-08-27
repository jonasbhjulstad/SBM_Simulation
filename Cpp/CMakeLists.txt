cmake_minimum_required(VERSION 3.5)
project(Eigen_FROLS)
find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)
find_package(SUNDIALS REQUIRED)
find_package(igraph REQUIRED)
set(SUNDIALS_LIBS SUNDIALS::cvode SUNDIALS::nvecserial)

file(GLOB casadi_shared_libs "/home/arch/Env/Default/lib/python3.10/site-packages/casadi/lib*.so")

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)


# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_executable(ex ex.cpp)
target_link_libraries(ex PRIVATE Eigen3::Eigen casadi)

add_subdirectory(test)
