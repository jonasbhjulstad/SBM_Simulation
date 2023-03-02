
if(${SYCL_GRAPH_ENABLE_TESTING})
include(CTest)

function(add_custom_test test_name)
    add_custom_executable(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_subdirectory(Buffer)
endif()

add_custom_executable(sycl_device_info)
add_custom_executable(sycl_copy_profiling)
add_custom_executable(sycl_kernel_emit)
add_custom_executable(tinymt_kernel)
add_custom_executable(invariant_graph)
# add_custom_executable(buffer_assign_add buffer_assign_add.cpp)