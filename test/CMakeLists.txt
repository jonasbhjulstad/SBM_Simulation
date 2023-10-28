cmake_minimum_required(VERSION 3.14...3.22)
set(PARENT_PROJECT_NAME ${PROJECT_NAME})

project(${PARENT_PROJECT_NAME}Tests LANGUAGES CXX)
if (NOT TARGET ${PROJECT_NAME})
# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

include(../cmake/tools.cmake)
#packageproject()
CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
CPMAddPackage("gh:doctest/doctest@2.4.11")
if(TEST_INSTALLED_VERSION)
find_package(${PARENT_PROJECT_NAME} REQUIRED)
else()
CPMAddPackage(NAME ${PARENT_PROJECT_NAME} SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)
endif()


# ---- Add SBM_DatabaseTests ----

enable_testing()

# Note: doctest and similar testing frameworks can automatically configure CMake tests. For other
# testing frameworks add the tests target instead: add_test(NAME ${PROJECT_NAME} COMMAND
# ${PROJECT_NAME})

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
add_subdirectory(source)


# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  target_compile_options(${PROJECT_NAME} PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(${PROJECT_NAME} PUBLIC -fprofile-arcs -ftest-coverage)
endif()
endif()