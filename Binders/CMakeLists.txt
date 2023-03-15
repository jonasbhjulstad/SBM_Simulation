# enable fPIC for custom_binder

function(add_custom_binder source_file)
    pybind11_add_module(${source_file} "${source_file}.cpp")
    target_link_libraries(${source_file} PUBLIC Sycl_Graph Static_RNG Tracy::TracyClient)
    target_compile_options(${source_file} PUBLIC ${SYCL_COMPILE_OPTIONS})
    target_compile_options(${source_file} PUBLIC ${DEFAULT_WARNING_FLAGS})
    target_compile_options(${source_file} PRIVATE ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -fPIC)
    # -fsycl-targets=intel_gpu_11_0_0)

    if(${SYCL_GRAPH_USE_CUDA})
        target_compile_options(${source_file} PRIVATE ${SYCL_CUDA_FLAGS})
        target_link_options(${source_file} PRIVATE ${SYCL_CUDA_FLAGS})
    endif()

    target_link_options(${source_file} PRIVATE ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -fPIC)
endfunction()

add_custom_binder(SBM_Binder)