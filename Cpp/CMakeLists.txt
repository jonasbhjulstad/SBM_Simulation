set(ENABLE_TESTING ON)
cmake_minimum_required(VERSION 3.5)
project(FROLS)
if(WIN32)
set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
set(PYTHON_EXECUTABLE "/home/arch/Env/Default/bin/python")
endif()

find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)
find_package(SUNDIALS REQUIRED)
find_package(pybind11 CONFIG)
find_package(OpenMP)
find_package(nlohmann_json REQUIRED)
if (UNIX)
set(CMAKE_LINKER "/usr/bin/lld")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(SUNDIALS_LIBS SUNDIALS::cvode SUNDIALS::nvecserial)
set(DEFAULT_LIBS Eigen3::Eigen ${SUNDIALS_LIBS} igraph::igraph)
set(CMAKE_CXX_FLAGS "-O0 -Wfatal-errors -fPIC")
include(FetchContent)
if (ENABLE_TESTING)
FetchContent_Declare(
googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)
endif()

# option(cppitertools_INSTALL_CMAKE_DIR "share/cppitertools/cmake")
# find_package(cppitertools QUIET)


FetchContent_Declare(
cppitertools_repo
GIT_REPOSITORY https://github.com/ryanhaining/cppitertools.git
GIT_TAG master
CONFIGURE_COMMAND ""
BUILD_COMMAND ""
)

FetchContent_MakeAvailable(cppitertools_repo)
# target_include_directories(cppitertools INTERFACE ${cppitertools_repo_SOURCE_DIR}/)
# find_package(cppitertools REQUIRED HINTS "${cppitertools_repo_BINARY_DIR}")


# find_package(sciplot REQUIRED)

find_package(igraph REQUIRED)
#set(FROLS_DATA_DIR "${PROJECT_SOURCE_DIR}/data" CACHE INTERNAL "DATA_DIR")
#set(FROLS_INCLUDE "${PROJECT_SOURCE_DIR}/include" CACHE INTERNAL "INCLUDE")
#set(FROLS_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/static;${PROJECT_SOURCE_DIR}/static/Features;${PROJECT_SOURCE_DIR}/static/Algorithm;${PROJECT_SOURCE_DIR}/static/Regressors" CACHE INTERNAL "INCLUDE_DIRS")

add_subdirectory(include)
add_subdirectory(static)
set(FROLS_LIBRARIES ${LIBRARIES} DataFrame FROLS_Eigen Regressors Features)
add_subdirectory(Binders)
configure_file("${PROJECT_SOURCE_DIR}/include/Path_Config.hpp.in" "${PROJECT_SOURCE_DIR}/include/Path_Config.hpp")
# For Windows: Prevent overriding the parent project's compiler/linker settings
if (ENABLE_TESTING)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(test)
message(STATUS "Testing enabled")
endif()
add_subdirectory(Executables)