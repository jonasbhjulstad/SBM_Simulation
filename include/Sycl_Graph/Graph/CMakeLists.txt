add_library(Graph_Base INTERFACE)
target_link_libraries(Graph_Base INTERFACE Common)
target_include_directories(Graph_Base INTERFACE ${SYCL_GRAPH_INCLUDE_DIRS})

add_subdirectory(Dynamic)
add_subdirectory(Sycl)

add_library(Graph INTERFACE)
target_link_libraries(Graph INTERFACE Graph_Base Graph_Sycl Graph_Dynamic)
# file(GLOB_RECURSE GRAPH_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
# if (${ENABLE_PRECOMPILED_HEADERS})
#     target_precompile_headers(Graph INTERFACE ${GRAPH_HEADER_FILES} REUSE_FROM Common)
# endif()