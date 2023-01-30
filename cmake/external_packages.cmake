include(cmake/CPM.cmake)

CPMFindPackage(
    NAME Static_RNG
    GITHUB_REPOSITORY jonasbhjulstad/Static_RNG
    GIT_TAG master
    OPTIONS
    "CMAKE_ARGS -DENABLE_SYCL=ON"
)
set(cppitertools_INSTALL_CMAKE_DIR share)
CPMFindPackage(
    NAME cppitertools
    GITHUB_REPOSITORY ryanhaining/cppitertools
    GIT_TAG master
)
include(FindThreads)
CPMFindPackage(
    NAME Tracy
    GITHUB_REPOSITORY wolfpld/tracy
    GIT_TAG master
)

CPMFindPackage(
    NAME Eigen3
    GITHUB_REPOSITORY libigl/eigen
    GIT_TAG master
    OPTIONS
    "QUIET ON"
)


#boost graph library
find_package(Boost REQUIRED COMPONENTS graph)