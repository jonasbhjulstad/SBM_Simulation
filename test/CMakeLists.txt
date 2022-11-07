
function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen Sycl_Graph ${ONEAPI_LIBRARIES})
endfunction()

add_custom_executable(array_constraints)
add_custom_executable(sycl_mutex)
# add_executable(csv_regression csv_regression.cpp)
# target_link_libraries(csv_regression PUBLIC Sycl_Graph Eigen3::Eigen ${ONEAPI_LIBRARIES} Regressors Graph_Math Features DataFrame Sycl_Graph_Eigen)