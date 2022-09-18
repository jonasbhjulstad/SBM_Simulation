include(GoogleTest)

function(add_custom_test source_file)
add_executable(${source_file} "${source_file}.cpp")
target_include_directories(${source_file} PUBLIC "../include")
target_link_libraries(${source_file} PUBLIC casadi Eigen3::Eigen GTest::gtest_main ${SUNDIALS_LIBS} ${LIBRARIES} graphlite FROLS_Graph)
gtest_discover_tests(${source_file})
endfunction()


# add_custom_test(SIR_integrator)
add_custom_test(monomials)
#add_custom_test(sir_bernoulli)
add_custom_test(casadi_function)
# add_custom_test(feature_transform)
