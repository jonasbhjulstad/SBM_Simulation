
function(default_configure_target target_name)
    target_link_libraries(${target_name} PUBLIC -ldl -lpq Static_RNG::Static_RNG)
    target_link_libraries(${target_name} PUBLIC TBB::tbb)
    target_compile_options(${target_name} PUBLIC -fsycl ${SYCL_CUSTOM_FLAGS} -fPIC)
    target_compile_options(${target_name} PUBLIC ${DEFAULT_WARNING_FLAGS} ${SBM_SIMULATION_DEFAULT_FLAGS})
    target_include_directories(${target_name} PUBLIC $<BUILD_INTERFACE:${SBM_SIMULATION_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>)
    target_link_options(${target_name} PUBLIC ${SYCL_CUSTOM_FLAGS} ${SBM_SIMULATION_DEFAULT_FLAGS})
    if (${ENABLE_CLANG_TIDY})
    set_target_properties(${target_name} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
    endif()
endfunction()

function(add_binder source_file)
    pybind11_add_module(${source_file} "${source_file}.cpp")
    default_configure_target(${source_file})
endfunction()

function(add_default_library source_file)
    add_library(${source_file} "${source_file}.cpp")
    default_configure_target(${source_file})
endfunction()


function(add_default_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")

endfunction()
