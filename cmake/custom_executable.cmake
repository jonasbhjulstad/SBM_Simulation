function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen Sycl_Graph ${ONEAPI_LIBRARIES} cppitertools Math tinymt)
    target_compile_options(${source_file} PUBLIC ${SYCL_COMPILE_OPTIONS})
    #link to sycl libraries
endfunction()