cmake_minimum_required(VERSION 3.14...3.22)

project(SBM_SimulationTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)
if(NOT ${IntelSYCL_FOUND})
find_package(IntelSYCL CONFIG REQUIRED)
endif()

option(ENABLE_CUDA "Enable CUDA" ON)
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
# add_link_options("-v")
if (${ENABLE_CUDA})
    # add_link_options(--cuda-path="/home/man/cuda_remote")
    # add_compile_options(--cuda-path="/home/man/cuda_remote")
    set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    set(SBM_SIMULATION_TESTS_COMPILE_DEFINITIONS ${SBM_SIMULATION_TESTS_COMPILE_DEFINITIONS} -DSBM_SIMULATION_TESTS_ENABLE_CUDA)
    # set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -fsycl-targets=nvidia_gpu_sm_86)
    # add_link_options(-fsycl -fsycl-targets=nvidia_gpu_sm_86)
    # add_compile_options(-Xsycl-target-backend=nvptx64-nvidia-cuda)
    # add_link_options(-Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86)
    # set(SYCL_CUSTOM_FLAGS ${SYCL_CUSTOM_FLAGS} -Xcuda-ptxas "--verbose")
endif()

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

include(${PROJECT_SOURCE_DIR}/../cmake/external_packages.cmake)
CPMAddPackage("gh:doctest/doctest@2.4.9")
CPMAddPackage("gh:TheLartians/Format.cmake@1.7.3")
if(TEST_INSTALLED_VERSION)
find_package(SBM_Simulation REQUIRED)
else()
CPMAddPackage(NAME SBM_Simulation SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)
endif()
file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
# ---- Create binary ----
add_executable(${PROJECT_NAME} ${sources})

target_link_libraries(${PROJECT_NAME} -ldl -lpq doctest::doctest SBM_Simulation::SBM_Simulation SOCI::soci_core SOCI::soci_postgresql)
target_compile_definitions(${PROJECT_NAME} PUBLIC ${SBM_SIMULATION_TESTS_COMPILE_DEFINITIONS})
target_link_options(${PROJECT_NAME} PUBLIC -fPIC)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # target_compile_options(SBM_Simulation PUBLIC -Wall -Wpedantic -Wextra -Werror)
  elseif(MSVC)
    target_compile_options(SBM_Simulation PUBLIC /W4 /WX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif()
endif()

# ---- Add SBM_SimulationTests ----

enable_testing()

# Note: doctest and similar testing frameworks can automatically configure CMake tests. For other
# testing frameworks add the tests target instead: add_test(NAME ${PROJECT_NAME} COMMAND
# ${PROJECT_NAME})

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(${PROJECT_NAME})

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  target_compile_options(SBM_Simulation PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(SBM_Simulation PUBLIC -fprofile-arcs -ftest-coverage)
endif()
