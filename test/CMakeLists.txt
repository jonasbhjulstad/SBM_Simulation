cmake_minimum_required(VERSION 3.22)

project(SBM_GraphTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)


# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)
if(NOT ${IntelSYCL_FOUND})
find_package(IntelSYCL CONFIG REQUIRED)
endif()

include(${PROJECT_SOURCE_DIR}/../cmake/external_packages.cmake)
CPMAddPackage("gh:doctest/doctest@2.4.9")
CPMAddPackage("gh:TheLartians/Format.cmake@1.7.3")
if(TEST_INSTALLED_VERSION)
find_package(SBM_Graph REQUIRED)
else()
CPMAddPackage(NAME SBM_Graph SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)
endif()
file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
# ---- Create binary ----
add_executable(${PROJECT_NAME} ${sources})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_link_libraries(${PROJECT_NAME} -ldl -lpq doctest::doctest SBM_Graph::SBM_Graph ${TINY_ORM_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${SBM_GRAPH_TESTS_COMPILE_DEFINITIONS})
target_link_options(${PROJECT_NAME} PUBLIC -fPIC)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # target_compile_options(SBM_Graph PUBLIC -Wall -Wpedantic -Wextra -Werror)
  elseif(MSVC)
    target_compile_options(SBM_Graph PUBLIC /W4 /WX)
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
  target_compile_options(SBM_Graph PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(SBM_Graph PUBLIC -fprofile-arcs -ftest-coverage)
endif()
