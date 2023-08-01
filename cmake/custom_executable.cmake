function(add_sycl_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_link_libraries(${source_file} PUBLIC Sycl_Graph Static_RNG::Static_RNG TBB::tbb cppitertools)
    target_compile_options(${source_file} PUBLIC ${SYCL_COMPILE_OPTIONS})
    target_compile_options(${source_file} PUBLIC ${DEFAULT_WARNING_FLAGS})
    target_compile_options(${source_file} PRIVATE ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda)
    if(${SYCL_GRAPH_USE_CUDA})
        target_compile_options(${source_file} PRIVATE ${SYCL_CUDA_FLAGS})
        target_link_options(${source_file} PRIVATE ${SYCL_CUDA_FLAGS})
    endif()
    target_link_options(${source_file} PRIVATE ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda)
endfunction()

function(add_regression_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_link_libraries(${source_file} PUBLIC Static_RNG::Static_RNG TBB::tbb cppitertools Eigen3::Eigen ortools::ortools Sycl_Graph)
endfunction()
