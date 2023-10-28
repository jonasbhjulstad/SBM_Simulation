
function(default_configure_target target_name)
    target_compile_options(${target_name} PUBLIC -fsycl ${SYCL_CUSTOM_FLAGS} -Wno-deprecated_declarations)
    target_link_options(${target_name} PUBLIC -fsycl -fsycl-targets=${${PROJECT_NAME}_SYCL_TARGETS} -Wno-deprecated-declarations)

    target_include_directories(${target_name} PUBLIC $<BUILD_INTERFACE:${${PROJECT_NAME}_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>)
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
