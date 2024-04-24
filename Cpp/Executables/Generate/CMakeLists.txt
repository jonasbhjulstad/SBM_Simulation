function (link_generation_target name)
target_link_libraries(${name} PUBLIC ${FROLS_COMMON_LIBS} ${PYTHON_LIBRARIES} Regressors)
endfunction()

function(add_generation_executable exec_name)
    add_executable(${exec_name} ${exec_name}.cpp)
    target_include_directories(${exec_name} PUBLIC ${PYTHON_INCLUDE_DIRS})
    link_generation_target(${exec_name})
endfunction()

add_generation_executable(Bernoulli_SIR_MC)

add_generation_executable(SIR_Sine)
target_link_libraries(SIR_Sine PUBLIC ${SUNDIALS_LIBS})
add_generation_executable(SIR_Sine_Discrete)

add_generation_executable(SIR_Sine_Stochastic)
target_link_libraries(SIR_Sine PUBLIC ${SUNDIALS_LIBS})
