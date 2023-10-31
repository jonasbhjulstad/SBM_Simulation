cmake_minimum_required(VERSION VERSION 3.22...3.27 FATAL_ERROR)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Initialize variables
# ---

set(Tom_ns tom)
set(Tom_target tom)



# TinyORM CMake modules (needed to set the executable version and RC file on Windows)
list(APPEND CMAKE_MODULE_PATH "${TinyORM_SOURCE_DIR}/cmake/CommonModules")

# build tree
list(APPEND CMAKE_PREFIX_PATH "${TinyOrmBuildDir}")

# Initialize Project Version
# ---

include(TinyHelpers)
tiny_read_version(TINY_VERSION
    TINY_VERSION_MAJOR TINY_VERSION_MINOR TINY_VERSION_PATCH TINY_VERSION_TWEAK
    VERSION_HEADER "${TinyORM_SOURCE_DIR}/tom/include/tom/version.hpp"
    PREFIX TINYTOM
    HEADER_FOR "${Tom_ns}"
)

# Basic project
# ---
set(PARENT_PROJECT_NAME ${PROJECT_NAME})
set(PARENT_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(PARENT_PROJECT_BINARY_DIR ${PROJECT_BINARY_DIR})
project(${Tom_ns}
    DESCRIPTION "Tom console application for TinyORM C++ library"
    HOMEPAGE_URL "https://www.tinyorm.org"
    LANGUAGES CXX
    VERSION ${TINY_VERSION}
)
# Tom command-line application
# ---

add_executable(${Tom_target}
    tom_executable.cpp
)
add_executable(${Tom_ns}::${Tom_target} ALIAS ${Tom_target})

# Tom command-line application specific configuration
# ---

set_target_properties(${Tom_target}
    PROPERTIES
        C_VISIBILITY_PRESET "hidden"
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN YES
        VERSION ${PROJECT_VERSION}
)

target_include_directories(${Tom_target}
    PRIVATE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/database>"
)

# Tom command-line application defines
# ---

target_compile_definitions(${Tom_target}
    PRIVATE
        PROJECT_TOM
)

# Windows resource and manifest files
# ---
add_subdirectory(database)
# Find icons, tom/version.hpp, and Windows manifest file for MinGW
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    tiny_set_rc_flags("-I \"${TinyORM_SOURCE_DIR}/tom/resources\"")
endif()

include(TinyResourceAndManifest)
tiny_resource_and_manifest(${Tom_target}
    OUTPUT_DIR "${TINY_BUILD_GENDIR}/tmp/"
    RESOURCES_DIR "${TinyORM_SOURCE_DIR}/tom/resources"
)
set(TOM_EXECUTABLE_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(tom_config.hpp.in tom_config.hpp)
configure_file(${PARENT_PROJECT_SOURCE_DIR}/cmake/tom_env.sh.in ${PARENT_PROJECT_BINARY_DIR}/env.sh)
# Resolve and link dependencies
# ---

# Unconditional dependencies
target_link_libraries(${Tom_target}
    PRIVATE
        ${TINY_ORM_LIBRARIES}
)
target_include_directories(${Tom_target} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
