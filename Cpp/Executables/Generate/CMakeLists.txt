function (link_generation_target name)
target_link_libraries(${name} PUBLIC ${FROLS_COMMON_LIBS})
endfunction()

function(add_generation_executable exec_name)
    add_executable(${exec_name} ${exec_name}.cpp)
    link_generation_target(${exec_name})
endfunction()

add_generation_executable(Bernoulli_SIR_MC)
target_link_sycl(Bernoulli_SIR_MC)

if(${FROLS_USE_SYCL})
add_sycl_executable(Sycl_SIR_MC)
link_generation_target(Sycl_SIR_MC)
add_sycl_executable(Sycl_Single_SIR_MC)
link_generation_target(Sycl_Single_SIR_MC)
endif()
add_generation_executable(SIR_Sine)
target_link_libraries(SIR_Sine PUBLIC SUNDIALS::nvecserial SUNDIALS::cvode SUNDIALS::cvodes)
add_generation_executable(SIR_Sine_Discrete)

add_generation_executable(SIR_Sine_Stochastic)
target_link_libraries(SIR_Sine PUBLIC SUNDIALS::nvecserial SUNDIALS::cvode SUNDIALS::cvodes)
