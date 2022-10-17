if (FROLS_USE_INTEL_SYCL)

function(add_sycl_executable name)
add_executable(${name} ${name}.cpp)
target_add_compile_definitions(${name} PRIVATE FROLS_USE_INTEL_SYCL)
target_add_compile_definitions(${name} PRIVATE ${SPIR_FORMAT})
target_add_compile_options(${name} PRIVATE "-fsycl")
target_add_compile_definitions(${name} PRIVATE FROLS_USE_SYCL)
endfunction()

elseif(FROLS_USE_HIPSYCL)

function(add_sycl_executable name)
add_executable(${name} ${name}.cpp)
add_sycl_to_target(TARGET ${name} SOURCES ${name}.cpp)
if(WIN32)
target_add_definitions(${name} PRIVATE -D_USE_MATH_DEFINES)
endif()

target_add_compile_definitions(${name} PRIVATE FROLS_USE_HIPSYCL)
target_add_compile_definitions(${name} PRIVATE FROLS_USE_SYCL)
target_add_compile_options(${name} PRIVATE "-fsycl")

endfunction()
else()
function(add_sycl_executable name)


endfunction()
endif()