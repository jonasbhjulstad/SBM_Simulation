include(FetchContent)

find_package(cppitertools CONFIG REQUIRED)


FetchContent_Declare(
    tinymt_repo
    GIT_REPOSITORY https://github.com/tueda/tinymt-cpp.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(tinymt_repo)
include(FindThreads)
find_package(Tracy CONFIG REQUIRED)

#boost graph library
find_package(Boost REQUIRED COMPONENTS graph)