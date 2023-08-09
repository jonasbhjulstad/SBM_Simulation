include("/home/man/.CPM/cpm/CPM_0.36.0.cmake")
CPMAddPackage("NAME;tinymt;GITHUB_REPOSITORY;tueda/tinymt-cpp;GIT_TAG;master;OPTIONS;BUILD_TESTING OFF")
set(tinymt_FOUND TRUE)