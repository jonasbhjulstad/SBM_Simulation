include("/home/man/Documents/Sycl_Graph/build/cmake/CPM_0.36.0.cmake")
CPMAddPackage("NAME;Static_RNG;GITHUB_REPOSITORY;jonasbhjulstad/Static_RNG;GIT_TAG;master;OPTIONS;ENABLE_SYCL;OFF;BUILD_PYTHON_BINDERS;OFF;BUILD_DOCS;OFF")
set(Static_RNG_FOUND TRUE)