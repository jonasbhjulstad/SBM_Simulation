set(ENABLE_TESTING ON)
cmake_minimum_required(VERSION 3.5)
project(FROLS)
if(WIN32)
set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
set(PYTHON_EXECUTABLE "/home/arch/Env/Default/bin/python")
endif()
list(APPEND CMAKE_MODULE_PATH "/home/arch/Downloads/cmake-checks-cache/CMakeChecksCache/")
find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)
find_package(ZLIB REQUIRED)
find_package(re2 REQUIRED)
find_package(Cbc REQUIRED)
get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
find_package(ortools CONFIG REQUIRED)
message(WARNING ${ortools_DIR})
get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
list(REMOVE_ITEM importTargetsAfter ${importTargets})

message("${importTargetsAfter}")
find_package(SUNDIALS REQUIRED)
find_package(pybind11 CONFIG)
find_package(OpenMP)
add_compile_options("-fPIC")
# find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
if (UNIX)
set(CMAKE_LINKER "/usr/bin/lld")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0069 NEW) 
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(SUNDIALS_LIBS SUNDIALS::cvode SUNDIALS::nvecserial)
set(DEFAULT_LIBS "Eigen3::Eigen")
# set(CMAKE_CXX_FLAGS "-O0 -Wfatal-errors -fPIC")
include(FetchContent)
if (ENABLE_TESTING)
FetchContent_Declare(
googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG release-1.12.1
)
target_link_directories(ortools::ortools INTERFACE "/usr/local/lib/")
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

set(GRAPH_LITE_SUPPORT_DOT OFF)
FetchContent_Declare(graphlite_repo
        GIT_REPOSITORY https://github.com/haasdo95/graphlite.git
        GIT_TAG master)

FetchContent_MakeAvailable(graphlite_repo)
set_target_properties(graphlite
        PROPERTIES
        CXX_STANDARD 17)

#FetchContent_Declare(matplotlibcpp_repo
#        GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
#        GIT_TAG master)
#FetchContent_MakeAvailable(matplotlibcpp_repo)
#
#FetchContent_Declare(papilo_repo
#        GIT_REPOSITORY https://github.com/scipopt/papilo.git
#        GIT_TAG master)
#FetchContent_MakeAvailable(papilo_repo)
#set(PAPILO_DIR ${CMAKE_BUILD_DIR}/CMakeFiles/)
#FetchContent_Declare(soplex_repo
#        GIT_REPOSITORY https://github.com/scipopt/soplex.git
#        GIT_TAG master)
#FetchContent_MakeAvailable(soplex_repo)
#FetchContent_Declare(scip_repo
#        GIT_REPOSITORY https://github.com/scipopt/scip.git
#        GIT_TAG master)
#FetchContent_MakeAvailable(scip_repo)

#set(USE_SCIP OFF)
#set(USE_PDLP OFF)
#FetchContent_Declare(ortools_repo
#        GIT_REPOSITORY https://github.com/google/or-tools.git
#        GIT_TAG master)
#FetchContent_MakeAvailable(ortools_repo)



# find_package(sciplot REQUIRED)

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
if (ENABLE_TESTING)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(test)
message(STATUS "Testing enabled")
endif()
add_subdirectory(Executables)