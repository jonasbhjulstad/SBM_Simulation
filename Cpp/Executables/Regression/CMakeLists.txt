
function(add_regression source_file)
add_executable(${source_file} "${source_file}.cpp")
target_link_libraries(${source_file} PUBLIC Common Eigen3::Eigen ${ONEAPI_LIBS} Regressors FROLS_Math Features DataFrame FROLS_Eigen FROLS_Network)
endfunction()
if (${ortools_FOUND})
add_regression(Quantile_Response)
target_link_libraries(Quantile_Response PUBLIC ortools::ortools)
endif()
add_regression(ERR_Response)
target_link_sycl(ERR_Response)
#add_regression(Linear_System_Deterministic)
add_regression(Linear_System_Stochastic)
add_regression(Quantile_ERR_Response)
add_regression(Bernoulli_MC_Regression)



