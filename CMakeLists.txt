#default cmakelists
cmake_minimum_required(VERSION 3.22)

#c++20
set(CMAKE_CXX_STANDARD 20)
project(SIR_SBM)
set(SIR_SBM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)
set(SIR_SBM_LIBRARIES "")
add_compile_options("-fsycl")
add_compile_options("-fcolor-diagnostics")
option(SIR_SBM_USE_GPU "Use GPU" OFF)
if(${SIR_SBM_USE_GPU})
add_compile_definitions(SIR_SBM_USE_GPU)
add_compile_options("-fsycl-targets=nvptx64-nvidia-cuda")
add_link_options("-fsycl-targets=nvptx64-nvidia-cuda")
else()
add_compile_options("-fsycl-targets=spir64_x86_64")
add_link_options("-fsycl-targets=spir64_x86_64")
endif()


include(cmake/external_packages.cmake)
include(cmake/default_targets.cmake)
add_library(SIR_SBM INTERFACE)
target_include_directories(SIR_SBM INTERFACE ${SIR_SBM_INCLUDE_DIR})
add_subdirectory(test)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)