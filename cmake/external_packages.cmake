include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)


CPMFindPackage(
    NAME Static_RNG
    GITHUB_REPOSITORY jonasbhjulstad/Static_RNG
    GIT_TAG master
)
# find_package(Static_RNG REQUIRED)

CPMFindPackage(
    NAME tinymt
    GITHUB_REPOSITORY tueda/tinymt-cpp
    GIT_TAG master
    OPTIONS
    "BUILD_TESTING OFF"
    "POSITION_INDEPENDENT_CODE ON"
)
find_package(Boost 1.78 REQUIRED COMPONENTS date_time HINTS ${PYTHON_ENV_CMAKE_MODULE_DIR})

set(cppitertools_INSTALL_CMAKE_DIR share)
CPMFindPackage(
    NAME cppitertools
    GITHUB_REPOSITORY ryanhaining/cppitertools
    GIT_TAG master
    OPTIONS
    "cppitertools_INSTALL_CMAKE_DIR share"
)

# qt_standard_project_setup()
find_package(TinyOrm CONFIG REQUIRED)
#eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#add pqxx4
CPMFindPackage(NAME Dataframe
    GITHUB_REPOSITORY jonasbhjulstad/Dataframe
    GIT_TAG master)

CPMFindPackage(NAME Buffer_Routines
    GITHUB_REPOSITORY jonasbhjulstad/Sycl_Buffer_Routines
    GIT_TAG master)


CPMFindPackage(NAME SBM_Database
    GITHUB_REPOSITORY jonasbhjulstad/SBM_Database
    GIT_TAG master
    OPTIONS
    #-fPIC
    "CMAKE_POSITION_INDEPENDENT_CODE ON"
    )

find_package(TBB REQUIRED)
include(FindThreads)
# find_package(casadi REQUIRED HINTS "/home/man/mambaforge/envs/gt/lib/cmake/casadi")


CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
