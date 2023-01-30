add_library(Network INTERFACE SIR_Bernoulli/SIR_Bernoulli.hpp)
target_include_directories(Network INTERFACE ${SYCL_GRAPH_INCLUDE_DIRS})
target_link_libraries(Network INTERFACE Math cppitertools Graph ${ONEAPI_LIBRARIES})

if (${ENABLE_PRECOMPILED_HEADERS})
    target_precompile_headers(Network INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Network.hpp REUSE_FROM Graph Math)
endif()
