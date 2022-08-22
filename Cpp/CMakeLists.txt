cmake_minimum_required(VERSION 3.5)
project(Eigen_FROLS)
find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED HINTS "/home/arch/Env/Default/lib/python3.10/site-packages/casadi/cmake")

add_executable(ex ex.cpp)
target_link_libraries(ex PRIVATE Eigen3::Eigen casadi)
