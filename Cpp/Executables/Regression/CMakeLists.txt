
function(add_regression source_file)
add_executable(${source_file} "${source_file}.cpp")
target_link_libraries(${source_file} PUBLIC Common Eigen3::Eigen TBB::tbb Regressors FROLS_Math Features DataFrame FROLS_Eigen graphlite ortools::ortools)
endfunction()
add_regression(ERR_Response)
add_regression(Quantile_Response)
#add_regression(Linear_System_Deterministic)
add_regression(Linear_System_Stochastic)


