include(FetchContent)
FetchContent_Declare(
    cppitertools_repo
    GIT_REPOSITORY https://github.com/ryanhaining/cppitertools.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(cppitertools_repo)


FetchContent_Declare(
    tinymt_repo
    GIT_REPOSITORY https://github.com/tueda/tinymt-cpp.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(tinymt_repo)