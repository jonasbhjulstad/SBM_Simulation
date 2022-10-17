cmake_minimum_required (VERSION 3.5)

project (all-pairs-shortest-paths)



# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()
set(HIPSYCL_TARGETS "cuda;")
find_package(hipSYCL)
add_compile_options("-fsycl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17")

add_executable(apsp src/apsp.cpp)
add_sycl_to_target(TARGET apsp SOURCES apsp.cpp)
if(WIN32)
add_custom_target(run apsp.exe)
else()
add_custom_target(run ./apsp)
target_link_libraries(apsp PUBLIC hipSYCL::hipSYCL-rt)
endif()
