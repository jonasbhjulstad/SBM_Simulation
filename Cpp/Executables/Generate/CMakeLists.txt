function (link_generation_target name)
target_link_libraries(${name} PUBLIC DataFrame Common FROLS_Math FROLS_Eigen graphlite FROLS_Network ${ONEAPI_LIBS} ${HIPSYCL_LIBS})
endfunction()

function(add_generation_executable exec_name)
    add_executable(${exec_name} ${exec_name}.cpp)
    link_generation_target(${exec_name})
endfunction()


add_generation_executable(Bernoulli_SIR_MC)
add_sycl_executable(Sycl_SIR_MC)
link_generation_target(Sycl_SIR_MC)
add_sycl_executable(Sycl_Single_SIR_MC)
link_generation_target(Sycl_Single_SIR_MC)


add_generation_executable(SIR_Sine)
target_link_libraries(SIR_Sine PRIVATE SUNDIALS::nvecserial SUNDIALS::cvode)
add_generation_executable(SIR_Sine_Discrete)
