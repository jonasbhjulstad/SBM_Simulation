include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

CPMFindPackage(
    NAME Static_RNG
    GITHUB_REPOSITORY jonasbhjulstad/Static_RNG
    GIT_TAG master
    OPTIONS
    STATIC_RNG_ENABLE_SYCL ON
    BUILD_PYTHON_BINDERS OFF
    BUILD_DOCS OFF
)


set(cppitertools_INSTALL_CMAKE_DIR share)
CPMFindPackage(
    NAME cppitertools
    GITHUB_REPOSITORY ryanhaining/cppitertools
    GIT_TAG master
    OPTIONS
    "cppitertools_INSTALL_CMAKE_DIR share"
)
find_package(TBB REQUIRED)
include(FindThreads)
CPMFindPackage(
    NAME Tracy
    GITHUB_REPOSITORY wolfpld/tracy
    GIT_TAG master
)

find_package(TBB REQUIRED)

CPMFindPackage(
    NAME Eigen3
    GITHUB_REPOSITORY libigl/eigen
    GIT_TAG master
    OPTIONS
    "QUIET ON"
)
# CPMFindPackage(
#     NAME DataFrame
#     GITHUB_REPOSITORY hosseinmoein/DataFrame
#     GIT_TAG master
# )

CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
