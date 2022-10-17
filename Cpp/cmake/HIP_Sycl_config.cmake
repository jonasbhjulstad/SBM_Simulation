if(${FROLS_USE_HIPSYCL})
    find_package(HIPSYCL REQUIRED)

    if(NOT HIPSYCL_DEBUG_LEVEL)
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(HIPSYCL_DEBUG_LEVEL 3 CACHE INTEGER
                "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                FORCE)
        else()
            set(HIPSYCL_DEBUG_LEVEL 2 CACHE INTEGER
                "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                FORCE)
        endif()
    endif()

    # add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
    # Use add_definitions for now for older cmake versions
    cmake_policy(SET CMP0005 NEW)
    add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})
    add_compile_options(--hipsycl-targets='cuda')

    set(FROLS_USE_SYCL ON)

    # add_compile_definitions(__HIPSYCL_ENABLE_SYCL_TARGET__)
endif()