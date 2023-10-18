set(ENABLE_TESTING ON)
cmake_minimum_required(VERSION 3.5)
project(SBM_Simulation VERSION 1.0.0 LANGUAGES CXX)
set(PROJECT_NAMESPACE "SBM_Simulation")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)
#set CMP0079 NEW
cmake_policy(SET CMP0079 NEW)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
# add_link_options("-v")
set(SYCL_HEADER_INCLUDE_DIR "/opt/intel/oneapi/compiler/latest/linux/include/sycl")
if(WIN32)
    set(PYTHON_EXECUTABLE "C:\\Users\\jonas\\anaconda3\\envs\\Network_MPC\\python.exe")
else()
    set(PYTHON_EXECUTABLE "/home/man/.pyenv/versions/3.9.13/bin/python3")
    set(PYTHON_INCLUDE_DIR "/home/man/.pyenv/versions/3.9.13/include/python3.9")
    set(PYTHON_LIBS "/home/man/.pyenv/versions/3.9.13/libs/python3.9")
    set(Python3_ROOT_DIR "/home/man/.pyenv/versions/3.9.13/")
endif()
# list(APPEND CMAKE_MODULE_PATH "/home/arch/Downloads/cmake-checks-cache/CMakeChecksCache/")
set(SPIR_FORMAT __DEVICE_SPIR64__)

set(PYBIND11_PYTHON_VERSION "3.9")
# add_link_options("-v")
include(FindPython3)
find_package(Python3 COMPONENTS Interpreter Development)
# set(SYCL_CUSTOM_TARGETS "nvptx64-nvidia-cuda")
message(STATUS "Using SYCL Custom Targets: ${SYCL_CUSTOM_TARGETS}")
# set(SYCL_CUSTOM_TARGETS "x86_64")
# set(SYCL_CUSTOM_TARGETS "spir64_x86_64")
# set(SYCL_CUSTOM_TARGETS "native_cpu")
option(ENABLE_TBB "Enable TBB" OFF)
option(ENABLE_CUDA "Enable CUDA" ON)
add_compile_options(-Wunused)
add_compile_options(-fsycl)
# set(SYCL_CUSTOM_FLAGS -sycl-std=2020 -fsycl -Wno-linker-warnings -Rno-debug-disables-optimization -Wno-unknown-cuda-version)
# add_compile_options(-fsycl -fsycl-unnamed-lambda -g -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend '--cuda-gpu-arch=sm_70')
# add_link_options(-fsycl -fsycl-unnamed-lambda -g -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend '--cuda-gpu-arch=sm_70')
if (${ENABLE_CUDA})
    # add_link_options(--cuda-path="/home/man/cuda_remote")
    # add_compile_options(--cuda-path="/home/man/cuda_remote")
    # set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda)
    set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -fsycl-targets=nvidia_gpu_sm_86)
    add_link_options(-fsycl -fsycl-targets=nvidia_gpu_sm_86)
    # add_compile_options(-Xsycl-target-backend=nvptx64-nvidia-cuda)
    add_link_options(-Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86)
endif()
set(CMAKE_WARN_DEPRECATED OFF)
option(DEFAULT_WARNING_SUPPRESSION "Default Warning Suppression" ON)
find_package(IntelSYCL CONFIG REQUIRED)
find_package(ortools CONFIG REQUIRED)
set(PYTHON_ENV_CMAKE_MODULE_DIR "/home/man/mambaforge/envs/gt/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
include(cmake/external_packages.cmake)
include(cmake/custom_executable.cmake)
include(cmake/custom_binder.cmake)
# include(cmake/ccache.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ENABLE_CLANG_TIDY "Enable Clang Tidy" OFF)
# search for clang-tidy
if(${ENABLE_CLANG_TIDY})
find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
endif()
# setup clang-tidy command from executable + options
set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXE}" "-checks=misc-header-include-cycle")

#Sycl warning suppression flags
set(CMAKE_UNITY_BUILD OFF)
add_compile_options("-O0")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options("-stdlib=libstdc++")
endif()
# set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")
add_compile_options("-Wno-deprecated-declarations")
add_compile_options("-Wno-unknown-cuda-version")

# set(VERBOSE_WARNING_FLAGS -pedantic -pedantic-errors -Wextra -Waggregate-return -Wcast-align -Wcast-qual  -Wchar-subscripts  -Wcomment -Wconversion -Wdisabled-optimization -Wno-error -Wfloat-equal  -Wformat  -Wformat=2 -Wformat-nonliteral -Wformat-security -Wformat-y2k -Wimplicit  -Wimport  -Winit-self  -Winline -Winvalid-pch  -Wlong-long -Wmissing-braces -Wmissing-field-initializers -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn -Wpacked  -Wpadded -Wparentheses  -Wpointer-arith -Wredundant-decls -Wreturn-type -Wsequence-point  -Wshadow -Wno-sign-compare  -Wstack-protector -Wstrict-aliasing -Wstrict-aliasing=2 -Wswitch  -Wswitch-default -Wswitch-enum -Wtrigraphs  -Wuninitialized -Wunknown-pragmas  -Wunreachable-code -Wunused -Wunused-function  -Wunused-label  -Wunused-parameter -Wunused-value  -Wunused-variable  -Wvariadic-macros -Wvolatile-register-var -Wwrite-strings)
set(VERBOSE_WARNING_FLAGS -Wno-error -Wall -Wextra -pedantic-errors -Wconversion -Wsign-conversion)
option(ENABLE_VERBOSE_WARNINGS "Enable Verbose Warnings" OFF)
if(${ENABLE_VERBOSE_WARNINGS})
    add_compile_options(${VERBOSE_WARNING_FLAGS})
endif()

set(SBM_SIMULATION_DATA_DIR "${PROJECT_SOURCE_DIR}/data" CACHE INTERNAL "SBM_SIMULATION_DATA_DIR")
set(SBM_SIMULATION_LOG_DIR "${PROJECT_SOURCE_DIR}/log" CACHE INTERNAL "SBM_SIMULATION_LOG_DIR")
set(SBM_SIMULATION_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" CACHE INTERNAL "SBM_SIMULATION_INCLUDE_DIR")
file(MAKE_DIRECTORY ${SBM_SIMULATION_DATA_DIR}/SIR_sim)



configure_file("${PROJECT_SOURCE_DIR}/include/SBM_Simulation/Utils/path_config.hpp.in" "${PROJECT_SOURCE_DIR}/include/SBM_Simulation/Utils/path_config.hpp")

add_compile_options("-fcolor-diagnostics")
add_subdirectory(src)


add_subdirectory(test)
add_subdirectory(Binders)
add_subdirectory(Executables)

find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs_doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target(SBM_Simulation_docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)


packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES "Eigen3 3.4.0; SBM_Database 1.0.0; soci 4.2.0"
)
