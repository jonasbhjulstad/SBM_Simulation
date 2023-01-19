if(${SYCL_GRAPH_USE_CUDA})
    add_compile_options(-fsycl-targets=nvptx64-nvidia-cuda)
endif()