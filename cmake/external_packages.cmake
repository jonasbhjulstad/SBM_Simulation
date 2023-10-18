include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)



find_package(Static_RNG REQUIRED)

CPMFindPackage(
    NAME tinymt
    GITHUB_REPOSITORY tueda/tinymt-cpp
    GIT_TAG master
    OPTIONS
    "BUILD_TESTING OFF"
)
find_package(Boost 1.78 REQUIRED HINTS ${PYTHON_ENV_CMAKE_MODULE_DIR})

set(cppitertools_INSTALL_CMAKE_DIR share)
CPMFindPackage(
    NAME cppitertools
    GITHUB_REPOSITORY ryanhaining/cppitertools
    GIT_TAG master
    OPTIONS
    "cppitertools_INSTALL_CMAKE_DIR share"
)

#eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#add pqxx
CPMFindPackage(
    NAME Dataframe
    GITHUB_REPOSITORY jonasbhjulstad/Dataframe
    GIT_TAG master)


# CPMFindPackage(
#         NAME soci
#         GITHUB_REPOSITORY SOCI/soci
#         GIT_TAG master
#     )

find_package(SOCI REQUIRED COMPONENTS soci_core soci_postgresql)


CPMFindPackage(
    NAME SBM_Database
    GITHUB_REPOSITORY jonasbhjulstad/SBM_Database
    GIT_TAG master)


find_package(TBB REQUIRED)
include(FindThreads)
#     GIT_TAG master
# )
find_package(casadi REQUIRED HINTS "/home/man/mambaforge/envs/gt/lib/cmake/casadi")
# find_package(nlohmann_json 3.11.2 REQUIRED)

find_package(TBB REQUIRED)

CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
