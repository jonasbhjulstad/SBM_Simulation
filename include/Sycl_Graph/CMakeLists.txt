add_subdirectory(Graph)
add_library(Common STATIC path_config.cpp)
target_link_libraries(Common PRIVATE Eigen3::Eigen)
target_include_directories(Common INTERFACE "$<BUILD_INTERFACE:${SYCL_GRAPH_INCLUDE_DIRS}/..>")

file(GLOB SYCL_GRAPH_COMMON_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
if (${ENABLE_PRECOMPILED_HEADERS})
    target_precompile_headers(Common INTERFACE ${SYCL_GRAPH_COMMON_HEADERS})
endif()