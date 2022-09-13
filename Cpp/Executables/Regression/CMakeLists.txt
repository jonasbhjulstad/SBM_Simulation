
function(add_regression source_file)
add_executable(${source_file} "${source_file}.cpp")
target_link_libraries(${source_file} PUBLIC Common Eigen3::Eigen Regressors Math Features DataFrame FROLS_Eigen)
endfunction()
add_regression(Single_Response)
add_regression(ERR_Response)
add_regression(Quantile_Response)


