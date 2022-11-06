
function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen ${SUNDIALS_LIBS} FROLS_Graph)
endfunction()

function(add_custom_test source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen FROLS_Graph)
    gtest_discover_tests(${source_file})
    target_compile_options(${source_file} PRIVATE "-Wfatal-errors")
endfunction()

# add_executable(csv_regression csv_regression.cpp)
# target_link_libraries(csv_regression PUBLIC Common Eigen3::Eigen ${ONEAPI_LIBS} Regressors Sycl_Graph_Math Features DataFrame Sycl_Graph_Eigen)