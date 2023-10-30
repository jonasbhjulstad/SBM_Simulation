cmake_minimum_required(VERSION 3.14...3.22)

# ---- Project ----

# Note: update this to your new project's name and version
project(
  Template_Project
  VERSION 1.0
  LANGUAGES CXX
)
if (NOT TARGET ${PROJECT_NAME})

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)
#export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info

option(ENABLE_TESTING "Enable Test Builds" ON)
set(${PROJECT_NAME}_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
# ---- Add source files ----
include(cmake/CPM.cmake)
CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
if(NOT ${PROJECT_NAME}_EXTERNAL_PACKAGES)
  set(${PROJECT_NAME}_EXTERNAL_PACKAGES "" CACHE STRING "" FORCE)
endif()
include(cmake/external_packages.cmake)
add_subdirectory(source)
if(${ENABLE_TESTING} AND NOT TARGET ${PROJECT_NAME}Tests)
add_subdirectory(test)
endif()


string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)


packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES ""
)
endif()
