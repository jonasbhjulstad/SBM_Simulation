add_compile_options("-fPIC")
add_compile_options("-fsycl")
#if debug mode is enabled, disable optimizations
if (${DEFAULT_WARNING_SUPPRESSION})
    add_compile_options("-Wno-deprecated-declarations")
    add_compile_options("-Rno-debug-disables-optimization")
endif()