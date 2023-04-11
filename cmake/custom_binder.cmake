function(add_custom_binder source_file)
    pybind11_add_module(${source_file} "${source_file}.cpp")
    target_link_directories(${source_file} PUBLIC ${Boost_LIBRARY_DIR_DEBUG} ${graph_tool_LIBRARY_DIR})
    target_link_libraries(${source_file} PUBLIC Sycl_Graph Static_RNG Tracy::TracyClient TBB::tbb Eigen3::Eigen ${Boost_LIBRARIES} graph_tool_core)
    target_compile_options(${source_file} PUBLIC ${SYCL_COMPILE_OPTIONS})
    target_compile_options(${source_file} PUBLIC "-Wdeprecated-declarations")
    target_compile_options(${source_file} PUBLIC ${DEFAULT_WARNING_FLAGS})
    target_compile_options(${source_file} PUBLIC ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -fPIC)
    target_include_directories(${source_file} PUBLIC ${SYCL_GRAPH_INCLUDE_DIR} ${graph_tool_INCLUDE_DIRS})
    if(${SYCL_GRAPH_USE_CUDA})
        target_compile_options(${source_file} PUBLIC ${SYCL_CUDA_FLAGS})
        target_link_options(${source_file} PUBLIC ${SYCL_CUDA_FLAGS})
    endif()

    target_link_options(${source_file} PUBLIC ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -fPIC)
endfunction()


function(add_custom_library source_file)
    add_library(${source_file} SHARED "${source_file}.cpp")
    target_link_directories(${source_file} PUBLIC ${Boost_LIBRARY_DIR_DEBUG})
    target_link_libraries(${source_file} PUBLIC Sycl_Graph Static_RNG Tracy::TracyClient TBB::tbb Eigen3::Eigen ${Boost_LIBRARIES})
    target_compile_options(${source_file} PUBLIC ${SYCL_COMPILE_OPTIONS})
    target_compile_options(${source_file} PUBLIC "-Wdeprecated-declarations")
    target_compile_options(${source_file} PUBLIC ${DEFAULT_WARNING_FLAGS})
    target_compile_options(${source_file} PUBLIC ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -fPIC)
    target_include_directories(${source_file} PUBLIC ${SYCL_GRAPH_INCLUDE_DIR} ${graph_tool_INCLUDE_DIRS})
    if(${SYCL_GRAPH_USE_CUDA})
        target_compile_options(${source_file} PUBLIC ${SYCL_CUDA_FLAGS})
        target_link_options(${source_file} PUBLIC ${SYCL_CUDA_FLAGS})
    endif()

    target_link_options(${source_file} PUBLIC ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -fPIC)
endfunction()