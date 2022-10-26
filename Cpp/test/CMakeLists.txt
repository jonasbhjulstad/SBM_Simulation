include(GoogleTest)

function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen ${SUNDIALS_LIBS} FROLS_Graph)
endfunction()

function(add_custom_test source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen ${SUNDIALS_LIBS} FROLS_Graph GTest::gtest_main)
    gtest_discover_tests(${source_file})
    target_compile_options(${source_file} PRIVATE "-Wfatal-errors")
endfunction()

add_executable(csv_regression csv_regression.cpp)
target_link_libraries(csv_regression PUBLIC Common Eigen3::Eigen ${ONEAPI_LIBS} Regressors FROLS_Math Features DataFrame FROLS_Eigen)

function(add_regression source_file)
add_executable(${source_file} "${source_file}.cpp")
target_link_libraries(${source_file} PUBLIC Common Eigen3::Eigen ${ONEAPI_LIBS} Regressors FROLS_Math Features DataFrame FROLS_Eigen FROLS_Network)
endfunction()
add_regression(polynomial_features)
add_regression(MC_SIR_test)
target_include_directories(MC_SIR_test PUBLIC "../Executables/Generate/")
# add_sycl_executable(sycl_info)
# target_link_libraries(sycl_info PUBLIC FROLS_Sycl)

# add_custom_executable(neighbor_test)
# #link frols_network
# target_link_libraries(neighbor_test PUBLIC FROLS_Graph FROLS_Network)
# target_include_directories(neighbor_test PUBLIC "../Executables/Generate/")
# add_custom_executable(sycl_info)


# add_custom_test(Graph_Iterators)
# target_include_directories(Graph_Iterators PUBLIC "../static/Graph/")
# add_custom_test(SIR_integrator)
# add_custom_test(monomials)
#add_custom_test(sir_bernoulli)
# add_custom_test(feature_transform)
