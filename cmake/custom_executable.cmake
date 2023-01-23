function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include"  ${SYCL_GRAPH_INCLUDE_DIRS})
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen Sycl_Graph ${ONEAPI_LIBRARIES} Math tinymt cppitertools::cppitertools Tracy::TracyClient ${SYCL_GRAPH_DEBUG_LIBRARIES} ${SYCL_GRAPH_LIBRARIES})
    target_compile_options(${source_file} PUBLIC ${SYCL_COMPILE_OPTIONS})
    target_compile_options(${source_file} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda>)
    target_link_options(${source_file} PRIVATE ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda)
    #link to sycl libraries
endfunction()