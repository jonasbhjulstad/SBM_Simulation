if (SYCL_GRAPH_USE_ONEAPI)

function(target_link_sycl name)
target_compile_definitions(${name} PRIVATE SYCL_GRAPH_USE_ONEAPI)
target_compile_definitions(${name} PRIVATE ${SPIR_FORMAT})
target_compile_options(${name} PRIVATE "-fsycl")
target_compile_definitions(${name} PRIVATE SYCL_GRAPH_USE_SYCL)
endfunction()



elseif(SYCL_GRAPH_USE_HIPSYCL)

function(target_link_sycl name)
add_sycl_to_target(TARGET ${name} SOURCES ${name}.cpp)
if(WIN32)
target_add_definitions(${name} PRIVATE -D_USE_MATH_DEFINES)
endif()

target_compile_definitions(${name} PRIVATE SYCL_GRAPH_USE_HIPSYCL)
target_compile_definitions(${name} PRIVATE SYCL_GRAPH_USE_SYCL)
target_compile_options(${name} PRIVATE "-fsycl")
target_link_libraries(${name} PUBLIC hipSYCL::hipSYCL-rt)
endfunction()

else()
function(target_link_sycl name)


endfunction()
endif()
function(add_sycl_executable name)
if(${SYCL_GRAPH_USE_SYCL})
add_executable(${name} ${name}.cpp)
target_link_sycl(${name})
endif()
endfunction()